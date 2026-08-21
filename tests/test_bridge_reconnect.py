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

import pytest

from flipper_mcp import bridge as bridge_module
from flipper_mcp.bridge import FlipperBridge, FlipperError
from tests.conftest import FakeSerial


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
