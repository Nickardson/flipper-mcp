"""Minimal protobuf RPC client for the Flipper Zero.

The Flipper CLI has no screenshot command. Screen frames are only reachable
through the protobuf RPC channel: write ``start_rpc_session`` on the CLI and
the port stops being line-oriented text and starts carrying length-delimited
``PB.Main`` messages until a ``StopSession`` brings it back.

Why hand-rolled instead of the ``protobuf`` runtime plus generated bindings:
we need exactly three request shapes and one response field. That is ~60
lines of varint plumbing, against a dependency whose generated modules would
have to be regenerated every time the firmware's .proto files move.

Field numbers below are transcribed from lab.flipper.net's compiled
descriptors (frontend/src/shared/lib/flipperJs/protobufCompiled.js), which is
the same wire contract the official web app talks:

    Main.command_id                      = 1   varint
    Main.system_ping_request             = 5   message
    Main.stop_session                    = 19  message
    Main.gui_start_screen_stream_request = 20  message
    Main.gui_stop_screen_stream_request  = 21  message
    Main.gui_screen_frame                = 22  message
    Gui.ScreenFrame.data                 = 1   bytes
    Gui.ScreenFrame.orientation          = 2   varint

RPC mode takes the port over completely, so every other tool in this server
is inoperable while a session is open. ``rpc_session`` is therefore a context
manager that always closes the session and resyncs the CLI on the way out,
including on exceptions.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional

from .bridge import FlipperBridge, FlipperError

WIRE_VARINT = 0
WIRE_LEN = 2

FIELD_COMMAND_ID = 1
FIELD_SYSTEM_PING_REQUEST = 5
FIELD_SYSTEM_PING_RESPONSE = 6
FIELD_STOP_SESSION = 19
FIELD_GUI_START_SCREEN_STREAM = 20
FIELD_GUI_STOP_SCREEN_STREAM = 21
FIELD_GUI_SCREEN_FRAME = 22

FIELD_FRAME_DATA = 1
FIELD_FRAME_ORIENTATION = 2


# -- wire primitives --------------------------------------------------------


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Return (value, new_pos). Raises IndexError on a truncated varint."""
    value = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _tag(field: int, wire: int) -> bytes:
    return _varint(field << 3 | wire)


def _submessage(field: int, payload: bytes) -> bytes:
    return _tag(field, WIRE_LEN) + _varint(len(payload)) + payload


def _fields(payload: bytes) -> Iterator[tuple[int, int, object]]:
    """Walk a protobuf payload, yielding (field_number, wire_type, value).

    Value is an int for varints and bytes for length-delimited fields.
    Unknown wire types abort the walk rather than guessing at lengths.
    """
    pos = 0
    end = len(payload)
    while pos < end:
        key, pos = _read_varint(payload, pos)
        field, wire = key >> 3, key & 0x07
        if wire == WIRE_VARINT:
            value, pos = _read_varint(payload, pos)
            yield field, wire, value
        elif wire == WIRE_LEN:
            length, pos = _read_varint(payload, pos)
            yield field, wire, payload[pos : pos + length]
            pos += length
        elif wire == 5:  # fixed32
            yield field, wire, payload[pos : pos + 4]
            pos += 4
        elif wire == 1:  # fixed64
            yield field, wire, payload[pos : pos + 8]
            pos += 8
        else:
            return


def encode_request(command_id: int, field: int, payload: bytes = b"") -> bytes:
    """Build one length-delimited ``Main`` carrying a single empty request."""
    body = _tag(FIELD_COMMAND_ID, WIRE_VARINT) + _varint(command_id)
    body += _submessage(field, payload)
    return _varint(len(body)) + body


def split_messages(buf: bytes) -> tuple[list[bytes], int]:
    """Split a raw byte stream into complete ``Main`` payloads.

    Returns (payloads, consumed) so the caller can keep the trailing partial
    message in its accumulator and retry once more bytes land.
    """
    payloads: list[bytes] = []
    pos = 0
    while pos < len(buf):
        try:
            length, start = _read_varint(buf, pos)
        except (IndexError, ValueError):
            break
        if length == 0 or start + length > len(buf):
            break
        payloads.append(buf[start : start + length])
        pos = start + length
    return payloads, pos


def extract(payload: bytes, field: int) -> Optional[bytes]:
    """Return the sub-message body carried in `field`, if the Main has one."""
    for got_field, wire, value in _fields(payload):
        if got_field == field and wire == WIRE_LEN:
            return value  # type: ignore[return-value]
    return None


def screen_frame(payload: bytes) -> Optional[tuple[bytes, int]]:
    """Pull (framebuffer, orientation) out of a Main, if it carries a frame."""
    inner = extract(payload, FIELD_GUI_SCREEN_FRAME)
    if inner is None:
        return None
    data = b""
    orientation = 0
    for sub_field, sub_wire, sub_value in _fields(inner):
        if sub_field == FIELD_FRAME_DATA and sub_wire == WIRE_LEN:
            data = sub_value  # type: ignore[assignment]
        elif sub_field == FIELD_FRAME_ORIENTATION and sub_wire == WIRE_VARINT:
            orientation = sub_value  # type: ignore[assignment]
    return data, orientation


# -- session ----------------------------------------------------------------


def _settle(bridge: FlipperBridge, quiet: float = 0.15, timeout: float = 1.5) -> bytes:
    """Swallow the CLI echo, waiting for the line to actually go quiet.

    A fixed sleep is a guess: too short and the tail of the echo lands in the
    binary stream, too long and every capture pays for it. Returns whatever
    was swallowed so a caller can report it when things go wrong.
    """
    seen = bytearray()
    last = time.monotonic()
    deadline = last + timeout
    while True:
        chunk = bridge.take_raw()
        now = time.monotonic()
        if chunk:
            seen.extend(chunk)
            last = now
        elif now - last >= quiet:
            return bytes(seen)
        if now >= deadline:
            return bytes(seen)
        time.sleep(0.02)


@contextmanager
def rpc_session(
    bridge: FlipperBridge, handshake_timeout: float = 2.0
) -> Iterator["RpcChannel"]:
    """Enter protobuf RPC mode for the duration of the block.

    Always tears the session down and resyncs the text CLI afterwards — a
    leaked session would leave every other tool talking protobuf at a parser
    that expects ANSI text.
    """
    bridge.drain()
    # Carriage return ONLY. The device switches to protobuf the instant it sees
    # the terminator, so a trailing newline would land in the binary stream,
    # where 0x0A reads as "next message is 10 bytes long" and eats the head of
    # the first real request. This costs an afternoon if you get it wrong.
    bridge.write_raw(b"start_rpc_session\r")
    # The CLI echoes the command and emits its last prompt before the switch.
    _settle(bridge)
    channel = RpcChannel(bridge)
    try:
        channel.handshake(timeout=handshake_timeout)
        yield channel
    finally:
        try:
            channel.send(FIELD_STOP_SESSION)
            time.sleep(0.2)
        except Exception:
            pass
        bridge.resync()


class RpcChannel:
    """Request/response plumbing over an open RPC session."""

    def __init__(self, bridge: FlipperBridge) -> None:
        self._bridge = bridge
        self._buf = bytearray()
        self._seen = bytearray()
        self._command_id = 0

    def send(self, field: int, payload: bytes = b"") -> int:
        self._command_id += 1
        self._bridge.write_raw(encode_request(self._command_id, field, payload))
        return self._command_id

    def handshake(self, timeout: float = 2.0) -> None:
        """Ping the device to confirm the port is really speaking protobuf.

        Without this a failed session switch surfaces much later as an
        inscrutable frame timeout. A failed ping instead reports what the
        device actually sent, which is usually plain CLI text.
        """
        self.send(FIELD_SYSTEM_PING_REQUEST)
        if self._await(FIELD_SYSTEM_PING_RESPONSE, timeout) is None:
            raise FlipperError(
                "The Flipper did not enter protobuf RPC mode "
                f"(no ping response in {timeout:g}s). "
                f"Received instead: {self.diagnostic()}"
            )

    def _await(self, field: int, timeout: float) -> Optional[bytes]:
        """Collect until a Main carrying `field` shows up, or time runs out."""
        deadline = time.monotonic() + timeout
        while True:
            chunk = self._bridge.take_raw()
            if chunk:
                self._buf.extend(chunk)
                self._seen.extend(chunk[: max(0, 256 - len(self._seen))])
            payloads, consumed = split_messages(bytes(self._buf))
            if consumed:
                del self._buf[:consumed]
            for payload in payloads:
                if extract(payload, field) is not None:
                    return payload
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.02)

    def await_frame(self, timeout: float = 3.0) -> tuple[bytes, int]:
        """Block until a ScreenFrame arrives; raise FlipperError on timeout."""
        payload = self._await(FIELD_GUI_SCREEN_FRAME, timeout)
        if payload is None:
            raise FlipperError(
                f"No screen frame arrived within {timeout:g}s, though RPC mode "
                f"was live. Received: {self.diagnostic()}"
            )
        frame = screen_frame(payload)
        assert frame is not None  # _await matched on this very field
        return frame

    def diagnostic(self) -> str:
        """A readable preview of the first bytes the device sent back."""
        if not self._seen:
            return "nothing at all"
        head = bytes(self._seen[:96])
        printable = "".join(chr(b) if 32 <= b < 127 else "." for b in head)
        return f"{len(self._seen)}+ bytes, hex={head[:32].hex()} ascii={printable!r}"
