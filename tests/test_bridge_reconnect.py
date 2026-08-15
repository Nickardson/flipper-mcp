"""Tests for the bridge's recovery from a stale serial handle.

The Flipper re-enumerates on USB whenever it reboots, is replugged, or leaves
protobuf RPC mode. The handle the bridge is holding survives that as an open
file object whose writes fail forever — on Windows with ERROR_BAD_COMMAND, on
POSIX with ENXIO. Since the MCP server caches a single bridge for its whole
lifetime, failing to notice would leave every tool broken until restart.

All tests run offline: ``serial.Serial`` is swapped for a fake whose writes can
be made to fail on demand, and auto-detection is stubbed out.
"""

from __future__ import annotations

import time

import pytest
import serial

from flipper_mcp import bridge as bridge_module
from flipper_mcp.bridge import FlipperBridge, FlipperError


class FakeSerial:
    """A serial port whose writes fail until the handle is reopened.

    ``failing`` models a handle that is stale from the outset. ``fail_after``
    models one that goes stale partway through an operation: that many writes
    succeed and every later one fails. Each instance is one *handle*, so a
    reconnect produces a fresh one — exactly the state change the bridge is
    supposed to bring about.
    """

    STALE = (
        "WriteFile failed (PermissionError(13, "
        "'The device does not recognize the command.', None, 22))"
    )

    def __init__(self, port, baudrate=115200, timeout=0.05, write_timeout=2.0):
        self.port = port
        self.timeout = timeout
        self.written = bytearray()
        self.failing = False
        self.fail_after = None
        self.closed = False
        self.reply = b">: \r\n"
        self._replied = False
        self._writes = 0

    def write(self, data: bytes) -> int:
        if self.closed:
            raise serial.SerialException("write on closed port")
        self._writes += 1
        if self.failing or (self.fail_after is not None and self._writes > self.fail_after):
            raise serial.SerialException(self.STALE)
        self.written.extend(data)
        self._replied = False
        return len(data)

    def flush(self) -> None:
        if self.failing:
            raise serial.SerialException("flush failed")

    def read(self, size: int = 1) -> bytes:
        # One reply per write, so the quiet-period detector sees traffic and
        # then silence rather than an endless stream. The empty-handed case
        # blocks for the read timeout exactly as pyserial does — returning
        # instantly would turn every reader thread into a hot spin, and the
        # daemon threads of bridges a test leaves open never stop.
        if self.closed:
            raise serial.SerialException("read on closed port")
        if self._replied:
            time.sleep(self.timeout)
            return b""
        self._replied = True
        return self.reply

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_ports(monkeypatch):
    """Install FakeSerial and record every handle the bridge opens."""
    handles: list[FakeSerial] = []

    def factory(port, **kwargs):
        handle = FakeSerial(port, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(bridge_module.serial, "Serial", factory)
    monkeypatch.setattr(FlipperBridge, "_auto_detect", staticmethod(lambda: "COM_TEST"))
    yield handles
    # Reader threads are daemons, so a bridge a test could not close (because
    # the call under test raised) would otherwise keep polling for the rest of
    # the session. Closing the handle ends its loop.
    for handle in handles:
        handle.close()


# -- the stale handle ------------------------------------------------------


def test_send_reconnects_after_a_stale_handle(fake_ports):
    bridge = FlipperBridge()
    assert bridge.send("device_info") == ""
    fake_ports[0].failing = True

    bridge.send("device_info")

    assert len(fake_ports) == 2, "expected a second handle to be opened"
    assert fake_ports[0].closed
    assert b"device_info" in bytes(fake_ports[1].written)
    bridge.close()


def test_reconnect_redetects_an_auto_detected_port(fake_ports, monkeypatch):
    """Windows hands out a new COM number when the device re-enumerates."""
    bridge = FlipperBridge()
    assert bridge.port == "COM_TEST"
    monkeypatch.setattr(FlipperBridge, "_auto_detect", staticmethod(lambda: "COM_MOVED"))
    fake_ports[0].failing = True

    bridge.send("device_info")

    assert bridge.port == "COM_MOVED"
    bridge.close()


def test_reconnect_keeps_an_explicitly_pinned_port(fake_ports):
    """FLIPPER_PORT is the user overriding detection; honour it on reconnect."""
    bridge = FlipperBridge(port="COM_PINNED")
    fake_ports[0].failing = True

    bridge.send("device_info")

    assert bridge.port == "COM_PINNED"
    assert fake_ports[1].port == "COM_PINNED"
    bridge.close()


def test_reconnect_starts_a_live_reader_thread(fake_ports):
    bridge = FlipperBridge()
    old_reader = bridge._reader
    fake_ports[0].failing = True

    bridge.send("device_info")

    assert bridge._reader is not old_reader
    assert bridge._reader.is_alive()
    assert not old_reader.is_alive(), "the previous reader should have been joined"
    bridge.close()


def test_a_dead_reader_thread_triggers_a_reconnect(fake_ports):
    """Reads can fail while writes still succeed; the reader exits on error.

    Writing to such a handle would strand the caller waiting on a buffer
    nobody is filling, so the bridge checks reader liveness before writing.
    """
    bridge = FlipperBridge()
    bridge._stop.set()
    bridge._reader.join(timeout=2.0)
    assert not bridge._reader.is_alive()

    bridge.send("device_info")

    assert len(fake_ports) == 2
    assert bridge._reader.is_alive()
    bridge.close()


# -- when recovery is not safe ---------------------------------------------


def test_a_mid_command_write_does_not_reconnect(fake_ports):
    """The second half of ``write_file`` must not land on a fresh connection.

    The file's contents are only meaningful to a device that already received
    ``storage write``; replaying them after a reconnect would feed the file
    to the CLI as commands.
    """
    bridge = FlipperBridge()

    # Let the handshake and `storage write` through, then go stale — so the
    # failure lands on the content, the write that must not be replayed.
    fake_ports[0].fail_after = fake_ports[0]._writes + 1

    with pytest.raises(FlipperError, match="mid-command"):
        bridge.write_file("/ext/test.txt", "payload")

    assert len(fake_ports) == 1, "no reconnect should have been attempted"
    bridge.close()


def test_raw_writes_do_not_reconnect_by_default(fake_ports):
    """RPC requests mid-session must fail loudly, not resurface on the CLI."""
    bridge = FlipperBridge()
    fake_ports[0].failing = True

    with pytest.raises(FlipperError, match="mid-command"):
        bridge.write_raw(b"\x01\x02")

    assert len(fake_ports) == 1
    bridge.close()


def test_raw_writes_reconnect_when_opening_a_session(fake_ports):
    bridge = FlipperBridge()
    fake_ports[0].failing = True

    bridge.write_raw(b"start_rpc_session\r", allow_reconnect=True)

    assert len(fake_ports) == 2
    assert bytes(fake_ports[1].written).endswith(b"start_rpc_session\r")
    bridge.close()


# -- when recovery fails ---------------------------------------------------


def test_an_unplugged_device_reports_the_detection_failure(fake_ports, monkeypatch):
    bridge = FlipperBridge()
    fake_ports[0].failing = True

    def gone() -> str:
        raise FlipperError("No Flipper Zero detected on Windows.")

    monkeypatch.setattr(FlipperBridge, "_auto_detect", staticmethod(gone))

    with pytest.raises(FlipperError, match="could not reconnect"):
        bridge.send("device_info")


def test_a_still_failing_port_after_reconnect_is_reported(fake_ports, monkeypatch):
    """A reconnect that opens but still cannot write must not loop."""
    bridge = FlipperBridge()

    def factory(port, **kwargs):
        handle = FakeSerial(port, **kwargs)
        handle.failing = True
        fake_ports.append(handle)
        return handle

    fake_ports[0].failing = True
    monkeypatch.setattr(bridge_module.serial, "Serial", factory)

    with pytest.raises(FlipperError):
        bridge.send("device_info")

    assert len(fake_ports) == 2, "exactly one reconnect attempt, no retry loop"
