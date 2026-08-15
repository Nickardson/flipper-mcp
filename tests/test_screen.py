"""Tests for the protobuf RPC codec and framebuffer/PNG pipeline.

All tests run offline. A fake bridge stands in for the serial port and
replays a canned ScreenFrame, so the full capture path is exercised without
hardware.

The wire-format assertions are pinned to the byte values emitted by
lab.flipper.net's compiled descriptors — if a firmware bump ever renumbers a
field, these fail loudly rather than silently timing out against a device.
"""

from __future__ import annotations

import hashlib
import struct
import zlib

import pytest

from flipper_mcp import rpc, screen
from flipper_mcp.bridge import FlipperError

# A recognizable framebuffer: deterministic, and not symmetric under rotation.
FRAME = bytes((x * 7 + (x >> 3)) & 0xFF for x in range(screen.FRAME_BYTES))


def make_frame_message(data: bytes = FRAME, orientation: int = 0) -> bytes:
    """Build the delimited Main a real Flipper pushes during a screen stream."""
    inner = rpc._submessage(rpc.FIELD_FRAME_DATA, data)
    if orientation:
        inner += rpc._tag(rpc.FIELD_FRAME_ORIENTATION, rpc.WIRE_VARINT) + rpc._varint(
            orientation
        )
    main = rpc._tag(rpc.FIELD_COMMAND_ID, rpc.WIRE_VARINT) + rpc._varint(1)
    main += rpc._submessage(rpc.FIELD_GUI_SCREEN_FRAME, inner)
    return rpc._varint(len(main)) + main


# -- wire format -----------------------------------------------------------


@pytest.mark.parametrize(
    "field,expected",
    [
        (rpc.FIELD_STOP_SESSION, b"\x9a\x01"),
        (rpc.FIELD_GUI_START_SCREEN_STREAM, b"\xa2\x01"),
        (rpc.FIELD_GUI_STOP_SCREEN_STREAM, b"\xaa\x01"),
        (rpc.FIELD_GUI_SCREEN_FRAME, b"\xb2\x01"),
    ],
)
def test_tags_match_official_descriptors(field: int, expected: bytes) -> None:
    assert rpc._tag(field, rpc.WIRE_LEN) == expected


@pytest.mark.parametrize("value", [0, 1, 127, 128, 300, 1024, 65535])
def test_varint_roundtrip(value: int) -> None:
    encoded = rpc._varint(value)
    assert rpc._read_varint(encoded, 0) == (value, len(encoded))


def test_encode_request_carries_command_id_and_empty_body() -> None:
    wire = rpc.encode_request(7, rpc.FIELD_GUI_START_SCREEN_STREAM)
    payloads, consumed = rpc.split_messages(wire)
    assert consumed == len(wire)
    fields = list(rpc._fields(payloads[0]))
    assert (rpc.FIELD_COMMAND_ID, rpc.WIRE_VARINT, 7) in fields
    assert (rpc.FIELD_GUI_START_SCREEN_STREAM, rpc.WIRE_LEN, b"") in fields


# -- framing ---------------------------------------------------------------


def test_split_holds_back_partial_message() -> None:
    message = make_frame_message()
    payloads, consumed = rpc.split_messages(message[:-10])
    assert payloads == []
    assert consumed == 0


def test_split_handles_back_to_back_messages() -> None:
    pair = make_frame_message() + make_frame_message()
    payloads, consumed = rpc.split_messages(pair + b"\x99")
    assert len(payloads) == 2
    assert consumed == len(pair)  # trailing partial byte left for the next read


def test_screen_frame_extracts_data_and_orientation() -> None:
    payloads, _ = rpc.split_messages(make_frame_message(orientation=1))
    data, orientation = rpc.screen_frame(payloads[0])
    assert data == FRAME
    assert orientation == 1


def test_screen_frame_ignores_unrelated_messages() -> None:
    wire = rpc.encode_request(1, rpc.FIELD_STOP_SESSION)
    payloads, _ = rpc.split_messages(wire)
    assert rpc.screen_frame(payloads[0]) is None


# -- framebuffer -----------------------------------------------------------


@pytest.mark.parametrize("x,y", [(0, 0), (5, 9), (64, 32), (127, 63)])
def test_unpack_matches_reference_renderer(x: int, y: int) -> None:
    """Mirror of frameRenderer.ts: i = (y >> 3) * 128 + x, bit = 1 << (y & 7)."""
    rows = screen.unpack(FRAME)
    assert rows[y][x] == bool(FRAME[(y >> 3) * 128 + x] & (1 << (y & 7)))


def test_unpack_rotates_180_for_flipped_orientation() -> None:
    normal = screen.unpack(FRAME, 0)
    flipped = screen.unpack(FRAME, 1)
    assert flipped[63][127] == normal[0][0]
    assert flipped[0][0] == normal[63][127]


def test_unpack_rejects_short_frame() -> None:
    with pytest.raises(FlipperError, match="Short framebuffer"):
        screen.unpack(FRAME[:512])


# -- PNG -------------------------------------------------------------------


def test_png_is_structurally_valid() -> None:
    png = screen.to_png(screen.unpack(FRAME), scale=4)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, color_type = struct.unpack(">IIBB", png[16:26])
    assert (width, height, depth, color_type) == (512, 256, 8, 0)

    pos, kinds = 8, []
    while pos < len(png):
        length = struct.unpack(">I", png[pos : pos + 4])[0]
        kind = png[pos + 4 : pos + 8]
        body = png[pos + 8 : pos + 8 + length]
        crc = struct.unpack(">I", png[pos + 8 + length : pos + 12 + length])[0]
        assert crc == zlib.crc32(kind + body), f"bad CRC on {kind!r}"
        kinds.append(kind.decode())
        pos += 12 + length
    assert kinds == ["IHDR", "IDAT", "IEND"]


def test_png_pixels_are_black_ink_on_white() -> None:
    rows = [[False] * screen.WIDTH for _ in range(screen.HEIGHT)]
    rows[0][0] = True
    png = screen.to_png(rows, scale=1)
    length = struct.unpack(">I", png[33:37])[0]
    raw = zlib.decompress(png[41 : 41 + length])
    assert raw[0] == 0  # filter byte
    assert raw[1] == screen.INK == 0
    assert raw[2] == screen.PAPER == 255


# -- capture ---------------------------------------------------------------


class FakeBridge:
    """Serial stand-in that answers `loader info`, pings, and one frame.

    ``answer_ping`` / ``answer_stream`` let a test simulate a device that
    never switched into RPC mode, or one that switched but never streamed.
    """

    def __init__(
        self,
        frame: bytes = FRAME,
        orientation: int = 0,
        answer_ping: bool = True,
        answer_stream: bool = True,
    ) -> None:
        self.written = bytearray()
        self.pending = bytearray()
        self._frame = frame
        self._orientation = orientation
        self._answer_ping = answer_ping
        self._answer_stream = answer_stream
        self.resynced = False

    def send(self, cmd: str, timeout: float = 10.0, quiet_ms: int = 300) -> str:
        return "app: Desktop" if cmd == "loader info" else ""

    def write_raw(self, data: bytes, allow_reconnect: bool = False) -> None:
        self.written.extend(data)
        payloads, _ = rpc.split_messages(bytes(data))
        for payload in payloads:
            for field, _wire, _value in rpc._fields(payload):
                if field == rpc.FIELD_SYSTEM_PING_REQUEST and self._answer_ping:
                    self.pending.extend(
                        rpc.encode_request(1, rpc.FIELD_SYSTEM_PING_RESPONSE)
                    )
                elif field == rpc.FIELD_GUI_START_SCREEN_STREAM and self._answer_stream:
                    self.pending.extend(
                        make_frame_message(self._frame, self._orientation)
                    )

    def take_raw(self) -> bytes:
        data = bytes(self.pending)
        self.pending.clear()
        return data

    def drain(self) -> None:
        self.pending.clear()

    def resync(self, timeout: float = 2.0) -> None:
        self.resynced = True


def test_session_opens_with_carriage_return_only() -> None:
    """A trailing \\n would land in the binary stream and desync the framing."""
    bridge = FakeBridge()
    screen.capture(bridge, scale=1)
    assert bridge.written.startswith(b"start_rpc_session\r")
    assert not bridge.written.startswith(b"start_rpc_session\r\n")


def test_capture_returns_png_and_envelope() -> None:
    bridge = FakeBridge()
    result = screen.capture(bridge, scale=2)

    assert result["png"][:8] == b"\x89PNG\r\n\x1a\n"
    envelope = result["envelope"]
    assert envelope["app"] == "Desktop"
    assert envelope["orientation"] == "horizontal"
    assert envelope["scale"] == 2
    assert envelope["text"] is None
    rows = screen.unpack(FRAME)
    assert envelope["frame_sha256"] == hashlib.sha256(screen.pack_rows(rows)).hexdigest()
    assert (
        envelope["body_sha256"]
        == hashlib.sha256(
            screen.pack_rows(rows[screen.STATUS_BAR_ROWS :])
        ).hexdigest()
    )


def test_capture_always_closes_the_session() -> None:
    """RPC mode must never leak — every other tool depends on the text CLI."""
    bridge = FakeBridge()
    screen.capture(bridge, scale=1)

    # The session opens with the literal CLI command; protobuf starts after it.
    prefix = b"start_rpc_session\r"
    assert bridge.written.startswith(prefix)
    payloads, _ = rpc.split_messages(bytes(bridge.written[len(prefix) :]))
    sent = [f for p in payloads for f, _w, _v in rpc._fields(p) if f != 1]
    assert rpc.FIELD_GUI_STOP_SCREEN_STREAM in sent
    assert rpc.FIELD_STOP_SESSION in sent
    assert sent.index(rpc.FIELD_GUI_STOP_SCREEN_STREAM) < sent.index(
        rpc.FIELD_STOP_SESSION
    )
    assert bridge.resynced


def test_capture_closes_session_even_when_no_frame_arrives() -> None:
    """RPC engaged but the stream stayed silent."""
    bridge = FakeBridge(answer_stream=False)

    with pytest.raises(FlipperError, match="No screen frame"):
        screen.capture(bridge, scale=1, timeout=0.2)
    assert bridge.resynced


def test_failed_session_switch_reports_what_arrived_instead() -> None:
    """A device still in CLI mode must not surface as a frame timeout."""
    bridge = FakeBridge(answer_ping=False)

    with pytest.raises(FlipperError, match="did not enter protobuf RPC mode"):
        screen.capture(bridge, scale=1, timeout=0.2)
    assert bridge.resynced


def _scribble(frame: bytes, row: int) -> bytes:
    """Flip every pixel on one display row, leaving the rest of the frame alone."""
    out = bytearray(frame)
    for x in range(screen.WIDTH):
        out[(row >> 3) * screen.WIDTH + x] ^= 1 << (row & 7)
    return bytes(out)


@pytest.mark.parametrize("row", [0, 7, 8, screen.STATUS_BAR_ROWS - 1])
def test_body_digest_ignores_every_status_bar_row(row: int) -> None:
    """The clock's lower rows sit past row 8 — a page-aligned crop leaked them."""
    a = screen.capture(FakeBridge(FRAME), scale=1)["envelope"]
    b = screen.capture(FakeBridge(_scribble(FRAME, row)), scale=1)["envelope"]
    assert a["body_sha256"] == b["body_sha256"]
    assert a["frame_sha256"] != b["frame_sha256"]


@pytest.mark.parametrize("row", [screen.STATUS_BAR_ROWS, 32, 63])
def test_body_digest_covers_content_rows(row: int) -> None:
    a = screen.capture(FakeBridge(FRAME), scale=1)["envelope"]
    b = screen.capture(FakeBridge(_scribble(FRAME, row)), scale=1)["envelope"]
    assert a["body_sha256"] != b["body_sha256"]


def test_pack_rows_is_dense_and_msb_first() -> None:
    rows = [[False] * screen.WIDTH for _ in range(2)]
    rows[0][0] = True
    rows[1][7] = True
    packed = screen.pack_rows(rows)
    assert len(packed) == 2 * screen.WIDTH // 8
    assert packed[0] == 0b10000000
    assert packed[screen.WIDTH // 8] == 0b00000001


@pytest.mark.parametrize("scale", [0, 9, -1])
def test_capture_rejects_absurd_scale(scale: int) -> None:
    with pytest.raises(FlipperError, match="scale must be"):
        screen.capture(FakeBridge(), scale=scale)
