"""Serial bridge to a Flipper Zero over USB-CDC.

Design notes:
  - Flipper exposes a line-oriented CLI over /dev/cu.usbmodemflip_* on macOS
    and /dev/serial/by-id/*Flipper* on Linux. Default framing is just CRLF.
  - The CLI echoes input and paints ANSI colors/escape codes. Both are
    stripped before returning output to the caller.
  - We don't rely solely on the `>: ` prompt to detect command completion.
    Some commands (subghz rx, ir rx, nfc detect) stream until interrupted,
    and others print the prompt mid-output. Instead we use a quiet-period
    detector: the command is "done" when the serial buffer has been silent
    for `quiet_ms` milliseconds. navcore's pi-flipper-hid uses the same
    trick and it's empirically robust.
  - Reads happen on a background thread that appends to a shared bytearray
    under a lock. This avoids blocking the main thread and lets us peek at
    buffer size for the quiet-period check.
  - A bridge outlives the connection it was built on. The Flipper re-enumerates
    on USB whenever it reboots, is replugged, or switches in and out of
    protobuf RPC mode, which leaves the open handle stale: on Windows every
    subsequent write fails with ERROR_BAD_COMMAND, on POSIX with ENXIO/EIO.
    Since the MCP server caches one bridge for its whole lifetime, a stale
    handle would otherwise brick every tool until the server restarted. Writes
    that begin an operation therefore reconnect and retry once — see
    ``reconnect``.
"""

from __future__ import annotations

import argparse
import glob
import os
import platform
import re
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

import serial
from serial.tools import list_ports

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[=>]|\x1b\][^\x07]*\x07")
PROMPT = ">:"

# Seconds of silence after which the bridge hands the port back to the OS.
# Set FLIPPER_IDLE_TIMEOUT=0 to keep it open for the life of the process.
DEFAULT_IDLE_TIMEOUT = float(os.environ.get("FLIPPER_IDLE_TIMEOUT", "120"))


class FlipperError(RuntimeError):
    pass


def _is_busy(error: Exception) -> bool:
    """Whether opening the port failed because someone else holds it.

    pyserial reports this differently per platform and only ever as a message
    string — macOS and Linux raise EBUSY ("Resource busy", errno 16), Windows
    raises ERROR_ACCESS_DENIED, which surfaces as "Access is denied". Matching
    only the POSIX wording, as an earlier version did, left Windows users
    reading a raw ctypes error for the most common failure there.

    This matters more now that the bridge parks the port when idle: sharing it
    with qFlipper or the web app makes "someone else has it" an ordinary
    outcome rather than a misconfiguration.
    """
    text = str(error)
    return any(
        marker in text
        for marker in ("Resource busy", "Errno 16", "Access is denied", "PermissionError(13")
    )


class FlipperBridge:
    """Thin wrapper around a Flipper Zero's USB-CDC CLI.

    Use ``send(cmd)`` for short one-shot commands that return a prompt.
    Use ``stream(cmd, duration)`` for commands that run continuously and
    need a Ctrl-C to stop (subghz rx, ir rx, nfc detect, etc).
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        read_timeout: float = 0.05,
        idle_timeout: Optional[float] = None,
    ) -> None:
        # Remember whether the port was pinned by the caller. On reconnect an
        # auto-detected port has to be detected again — Windows hands out a
        # different COM number when the device re-enumerates.
        self._pinned_port = port
        self._baudrate = baudrate
        self._read_timeout = read_timeout
        self._idle_timeout = (
            DEFAULT_IDLE_TIMEOUT if idle_timeout is None else idle_timeout
        )
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self._ser: Optional[serial.Serial] = None
        # Reentrant: an operation holds this for its whole duration, and the
        # calls nested inside it re-acquire freely. It is what makes the idle
        # sweep safe — the port can only be dropped between operations.
        self._state_lock = threading.RLock()
        self._last_used = time.monotonic()
        self._busy = 0
        self._shutdown = threading.Event()
        self._janitor: Optional[threading.Thread] = None
        self._open()
        self._start_janitor()

    # -- connection ---------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._ser is not None

    def _open(self) -> None:
        """Detect, open, and hand the port to a fresh reader thread."""
        self.port = self._pinned_port or self._auto_detect()
        try:
            self._ser = serial.Serial(
                self.port,
                baudrate=self._baudrate,
                timeout=self._read_timeout,
                write_timeout=2.0,
            )
        except serial.SerialException as e:
            if _is_busy(e):
                raise FlipperError(
                    f"Flipper port {self.port} is busy — another app has it open. "
                    "Common culprits: the 'Flipper Lab' tab in Chrome "
                    "(lab.flipper.net), qFlipper, an open `screen`/`tio` "
                    "session, or a second copy of this MCP server. Close it "
                    "and retry."
                ) from e
            raise FlipperError(f"Could not open {self.port}: {e}") from e
        self._stop.clear()
        self._drain()
        self._reader = threading.Thread(
            target=self._read_loop, name="flipper-reader", daemon=True
        )
        self._reader.start()
        self._handshake()

    def _teardown(self) -> None:
        """Stop the reader and drop the handle. Safe on an already-dead port."""
        self._stop.set()
        reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            # The read loop blocks for at most `read_timeout`, so this returns
            # promptly. Joining matters: two reader threads appending to the
            # same buffer would interleave old and new bytes.
            reader.join(timeout=2.0)
        handle, self._ser = self._ser, None
        try:
            if handle is not None:
                handle.close()
        except Exception:
            pass

    def reconnect(self) -> None:
        """Rebuild the connection after the device re-enumerated.

        Anything buffered belonged to the old connection and is discarded, so
        this is only safe between operations — never mid-command, and never
        inside an RPC session, where it would drop the CLI back in front of a
        caller still speaking protobuf.
        """
        with self._state_lock:
            self._teardown()
            self._open()

    def _ensure_open(self) -> None:
        """Reopen if the port was released, or if the connection went bad."""
        with self._state_lock:
            if self._ser is None:
                self._open()
            elif self._reader is None or not self._reader.is_alive():
                # A dead reader means the port failed on the read side — the
                # loop exits on error and never comes back. Writes can still
                # succeed against such a handle, which would strand the caller
                # waiting on a quiet period nobody is filling any more.
                self.reconnect()

    # -- idle release -------------------------------------------------------
    #
    # Holding the port open forever is antisocial: it is an exclusive handle on
    # Windows and macOS, so a parked MCP server locks qFlipper, the Flipper Lab
    # web app, and `screen`/`tio` out of the device for as long as it runs. The
    # connection is cheap to rebuild — detect, open, handshake, measured at
    # ~0.75s on Windows — so a bridge that has gone quiet drops the port and
    # reopens on the next command, which is the only one that pays.

    @contextmanager
    def _operation(self) -> Iterator[None]:
        """Hold the connection open for the whole of one operation.

        The port can only be reclaimed while this is not held, so no command
        can have it pulled out from under it midway.
        """
        with self._state_lock:
            self._ensure_open()
            self._busy += 1
            try:
                yield
            finally:
                self._busy -= 1
                self._last_used = time.monotonic()

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Keep the port open across a sequence of calls.

        Needed by anything that spans more than one bridge call and cannot
        survive a reconnect in the middle — an RPC session above all, where a
        reopened port would land the CLI in front of a caller still writing
        protobuf.
        """
        with self._operation():
            yield

    def _start_janitor(self) -> None:
        if self._idle_timeout <= 0:
            return  # explicitly disabled: hold the port for the whole session
        self._janitor = threading.Thread(
            target=self._janitor_loop, name="flipper-janitor", daemon=True
        )
        self._janitor.start()

    def _janitor_loop(self) -> None:
        # Quarter of the timeout bounds how long past the deadline the port can
        # linger, without waking often enough to matter. The floor only comes
        # into play for the very short timeouts the tests use.
        tick = max(0.25, self._idle_timeout / 4)
        while not self._shutdown.wait(tick):
            self.release_if_idle()

    def release_if_idle(self) -> bool:
        """Drop the port if nothing has used it for ``idle_timeout``.

        Returns whether the port was released. Never waits on a busy bridge:
        an operation in flight is itself proof the bridge is not idle, so the
        sweep skips this round rather than queueing behind it.
        """
        if self._idle_timeout <= 0:
            return False
        if not self._state_lock.acquire(blocking=False):
            return False
        try:
            # The lock alone would not settle this. It is reentrant, so a
            # caller that reached here from inside its own operation would
            # acquire it happily and release the port under itself. `_busy`
            # states the invariant outright instead of leaning on which thread
            # happens to be asking.
            if self._busy or self._ser is None:
                return False
            if time.monotonic() - self._last_used < self._idle_timeout:
                return False
            self._teardown()
            return True
        finally:
            self._state_lock.release()

    # -- discovery ----------------------------------------------------------

    @staticmethod
    def _auto_detect() -> str:
        sysname = platform.system()
        if sysname == "Darwin":
            candidates = sorted(glob.glob("/dev/cu.usbmodemflip_*"))
        elif sysname == "Linux":
            candidates = sorted(glob.glob("/dev/serial/by-id/*Flipper*"))
        elif sysname == "Windows":
            # Windows: Flipper enumerates as a generic "USB Serial Device"
            # (STMicro VCP driver) — the description gives us nothing useful.
            # Match on the actual VID:PID (0483:5740, STMicro's VCP ID that
            # Flipper uses) or the "FLIP_" prefix Flipper puts in its USB
            # serial number, e.g. hwid="USB VID:PID=0483:5740 SER=FLIP_AN9A1ITE ...".
            candidates = []
            for port_info in list_ports.comports():
                hwid = (port_info.hwid or "").upper()
                desc = (port_info.description or "").lower()
                if (
                    "VID:PID=0483:5740" in hwid
                    or "SER=FLIP_" in hwid
                    or "flipper" in desc
                ):
                    candidates.append(port_info.device)
            candidates = sorted(candidates)
        else:
            candidates = []
        if not candidates:
            raise FlipperError(
                f"No Flipper Zero detected on {sysname}. "
                "Plug in via USB-C, confirm the device is unlocked, "
                "or set FLIPPER_PORT to an explicit device path."
            )
        return candidates[0]

    # -- reader thread ------------------------------------------------------

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._ser.read(4096)
            except Exception:
                # Deliberately broad. A port that dies mid-read raises whatever
                # the platform backend happens to raise — pyserial's Windows
                # reader trips over its own torn-down OVERLAPPED struct and
                # throws TypeError, not SerialException. Any of them mean the
                # same thing: this connection is finished. Exit quietly and let
                # the next write notice the dead thread and reconnect, rather
                # than dumping a traceback into the MCP server's stderr.
                break
            if data:
                with self._lock:
                    self._buf.extend(data)

    def _drain(self) -> None:
        with self._lock:
            self._buf.clear()

    def _snapshot(self) -> bytes:
        with self._lock:
            data = bytes(self._buf)
            self._buf.clear()
        return data

    def _peek_size(self) -> int:
        with self._lock:
            return len(self._buf)

    # -- writing ------------------------------------------------------------
    #
    # `allow_reconnect` is opt-in per call site rather than automatic, because
    # a reconnect is only harmless on the *first* write of an operation. Retry
    # a later write in a sequence and the bytes land on a device that never saw
    # the opening command — `write_file` would spill a file's contents onto the
    # CLI as commands, and an RPC request would arrive at a device back in text
    # mode. First writes carry no such history, so those are the ones marked.

    def _raw_write(self, data: bytes, allow_reconnect: bool) -> None:
        if self._ser is None:
            # Only reachable if a caller reaches past the public API; every
            # operation opens the port before writing a byte.
            raise FlipperError(
                f"No open connection to the Flipper on {self.port}."
            )
        try:
            self._ser.write(data)
            self._ser.flush()
        except Exception as e:
            # Broad for the same reason as `_read_loop`: pyserial reports a
            # dead handle as SerialException on most paths and as TypeError
            # from the Windows backend. Narrowing to SerialException would let
            # precisely the Windows case escape unhealed.
            if not allow_reconnect:
                raise FlipperError(
                    f"Lost the connection to the Flipper on {self.port} "
                    f"mid-command: {e}. The command may have half-executed; "
                    "retry it once the device is back."
                ) from e
            try:
                self.reconnect()
            except FlipperError as reconnect_error:
                raise FlipperError(
                    f"Lost the connection to the Flipper on {self.port} ({e}) "
                    f"and could not reconnect: {reconnect_error}"
                ) from e
            try:
                self._ser.write(data)
                self._ser.flush()
            except Exception as retry_error:
                raise FlipperError(
                    f"Reconnected to the Flipper on {self.port} but the write "
                    f"still failed: {retry_error}"
                ) from retry_error

    def _write(self, payload: str, allow_reconnect: bool = False) -> None:
        self._raw_write(payload.encode("utf-8"), allow_reconnect)

    # -- waiting ------------------------------------------------------------

    def _wait_quiet(self, timeout: float, quiet_ms: int) -> None:
        """Block until the buffer has been silent for `quiet_ms`, or timeout."""
        quiet_s = quiet_ms / 1000.0
        deadline = time.monotonic() + timeout
        last_size = self._peek_size()
        last_change = time.monotonic()

        while time.monotonic() < deadline:
            time.sleep(0.02)
            size = self._peek_size()
            now = time.monotonic()
            if size != last_size:
                last_size = size
                last_change = now
            elif size > 0 and (now - last_change) >= quiet_s:
                return
        # Timed out — return whatever we have

    # -- cleanup ------------------------------------------------------------

    @staticmethod
    def _clean(raw: bytes, cmd: Optional[str] = None) -> str:
        text = raw.decode("utf-8", errors="replace")
        text = ANSI_RE.sub("", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip a leading echo of the command we just sent.
        if cmd:
            stripped_cmd = cmd.strip()
            lines = text.split("\n", 1)
            if lines and lines[0].strip() == stripped_cmd:
                text = lines[1] if len(lines) > 1 else ""

        # Strip trailing prompt (">:" with optional whitespace).
        text = text.rstrip()
        while text.endswith(PROMPT):
            text = text[: -len(PROMPT)].rstrip()
        return text

    # -- lifecycle ----------------------------------------------------------

    def _handshake(self) -> None:
        # macOS opens USB-CDC with DTR asserted, which triggers the Flipper's
        # welcome banner (ASCII dolphin + firmware string + prompt). If we
        # drain too early the banner leaks into the first command's output.
        # Wait for the line to go quiet, *then* drain.
        self._write("\r\n")
        self._wait_quiet(timeout=3.0, quiet_ms=500)
        self._drain()

    def close(self) -> None:
        self._shutdown.set()  # stop the idle sweep before dropping the port
        with self._state_lock:
            self._teardown()

    def __enter__(self) -> "FlipperBridge":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- public API ---------------------------------------------------------

    def send(self, cmd: str, timeout: float = 10.0, quiet_ms: int = 300) -> str:
        """Run a one-shot command, return cleaned output once it settles."""
        with self._operation():
            self._drain()
            self._write(cmd + "\r\n", allow_reconnect=True)
            self._wait_quiet(timeout, quiet_ms)
            return self._clean(self._snapshot(), cmd)

    def stream(
        self,
        cmd: str,
        duration: float,
        post_timeout: float = 5.0,
        quiet_ms: int = 300,
    ) -> str:
        """Run a streaming command for `duration` seconds, Ctrl-C, return output."""
        with self._operation():
            self._drain()
            self._write(cmd + "\r\n", allow_reconnect=True)
            time.sleep(duration)
            self._write("\x03")  # Ctrl-C
            self._wait_quiet(post_timeout, quiet_ms)
            return self._clean(self._snapshot(), cmd)

    def interrupt(self) -> str:
        """Send a lone Ctrl-C to recover from a stuck streaming command.

        Useful when an earlier call left the Flipper in a listening state
        (e.g. ``subghz rx_raw`` invoked via ``flipper_cli`` without a
        terminating Ctrl-C). Returns whatever output the CLI emits while
        closing out.
        """
        with self._operation():
            self._drain()
            self._write("\x03", allow_reconnect=True)
            self._wait_quiet(timeout=2.0, quiet_ms=200)
            return self._clean(self._snapshot())

    # -- raw binary access --------------------------------------------------
    #
    # The protobuf RPC mode (see rpc.py) speaks binary over the same port, so
    # it needs the buffer untouched — no ANSI stripping, no CRLF rewriting,
    # no prompt trimming. These three are the whole escape hatch.

    def write_raw(self, data: bytes, allow_reconnect: bool = False) -> None:
        """Write raw bytes verbatim — no encoding, no line terminator.

        Defaults to no reconnect: mid-session RPC requests must fail loudly
        rather than resurface on a device that has been reset back to the CLI.
        Only the write that opens a session may set ``allow_reconnect``.
        """
        self._raw_write(data, allow_reconnect)

    def take_raw(self) -> bytes:
        """Pop everything buffered so far, unprocessed."""
        return self._snapshot()

    def drain(self) -> None:
        """Discard anything buffered. Used to swallow CLI echo before RPC."""
        self._drain()

    def resync(self, timeout: float = 2.0) -> None:
        """Return the CLI to a known state after raw/binary traffic."""
        with self._operation():
            self._write("\r\n", allow_reconnect=True)
            self._wait_quiet(timeout, quiet_ms=200)
            self._drain()

    def write_file(
        self,
        path: str,
        content: str,
        end_timeout: float = 3.0,
    ) -> str:
        """Write text content to a file on the Flipper via the CLI.

        Uses ``storage write <path>`` which reads stdin until Ctrl-C. Small
        text files only — not suitable for binary payloads or large files.
        """
        with self._operation():
            self._drain()
            self._write(f"storage write {path}\r\n", allow_reconnect=True)
            # Give the Flipper a moment to open the file and start reading
            time.sleep(0.3)
            self._write(content)
            if not content.endswith("\n"):
                self._write("\n")
            self._write("\x03")  # end write session
            self._wait_quiet(timeout=end_timeout, quiet_ms=300)
            return self._clean(self._snapshot())


# ---------------------------------------------------------------------------
# Smoke test entry point — `flipper-smoke` or `python -m flipper_mcp.bridge`.
# ---------------------------------------------------------------------------


def smoke() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the Flipper USB bridge.")
    parser.add_argument(
        "--port",
        default=os.environ.get("FLIPPER_PORT"),
        help="Explicit serial device path. Default: auto-detect.",
    )
    parser.add_argument(
        "--cmd",
        default="device_info",
        help="CLI command to run. Default: device_info.",
    )
    args = parser.parse_args()

    try:
        bridge = FlipperBridge(port=args.port)
    except FlipperError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"connected: {bridge.port}")
    try:
        output = bridge.send(args.cmd)
    finally:
        bridge.close()

    print(f"--- output of `{args.cmd}` ---")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(smoke())
