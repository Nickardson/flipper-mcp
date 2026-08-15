"""Tests for handing the serial port back after a period of inactivity.

A serial handle is exclusive on Windows and macOS, so an MCP server that holds
one for its whole lifetime locks qFlipper, the Flipper Lab web app, and
`screen`/`tio` out of the device even while doing nothing. The bridge therefore
drops the port once it has gone quiet and reopens it on the next command.

The invariant that matters is that the release is invisible to callers: it may
only happen *between* operations, never inside one.
"""

from __future__ import annotations

import threading
import time

import pytest

from flipper_mcp.bridge import FlipperBridge

# Long enough that nothing expires mid-test; short enough to trip on demand.
NEVER = 3600.0


def stale(bridge: FlipperBridge) -> None:
    """Backdate the activity clock so the bridge reads as idle."""
    bridge._last_used -= bridge._idle_timeout + 1


# -- releasing -------------------------------------------------------------


def test_an_idle_port_is_released(fake_ports):
    bridge = FlipperBridge(idle_timeout=NEVER)
    stale(bridge)

    assert bridge.release_if_idle() is True
    assert not bridge.connected
    assert fake_ports[0].closed, "the OS should have the port back"
    bridge.close()


def test_a_recently_used_port_is_kept(fake_ports):
    bridge = FlipperBridge(idle_timeout=NEVER)
    bridge.send("device_info")

    assert bridge.release_if_idle() is False
    assert bridge.connected
    bridge.close()


def test_releasing_stops_the_reader_thread(fake_ports):
    """A released port must leave nothing behind polling it."""
    bridge = FlipperBridge(idle_timeout=NEVER)
    reader = bridge._reader
    stale(bridge)

    bridge.release_if_idle()

    assert not reader.is_alive()
    bridge.close()


def test_the_idle_sweep_runs_on_its_own(fake_ports):
    """The release has to happen unprompted — that is the whole point."""
    bridge = FlipperBridge(idle_timeout=0.2)

    deadline = time.monotonic() + 5.0
    while bridge.connected and time.monotonic() < deadline:
        time.sleep(0.05)

    assert not bridge.connected, "the janitor thread never released the port"
    bridge.close()


def test_release_can_be_disabled(fake_ports):
    """FLIPPER_IDLE_TIMEOUT=0 is the escape hatch for holding the port."""
    bridge = FlipperBridge(idle_timeout=0)
    stale(bridge)

    assert bridge.release_if_idle() is False
    assert bridge.connected
    assert bridge._janitor is None, "no sweep thread should have been started"
    bridge.close()


# -- reopening -------------------------------------------------------------


def test_the_next_command_reopens_the_port(fake_ports):
    bridge = FlipperBridge(idle_timeout=NEVER)
    stale(bridge)
    bridge.release_if_idle()

    bridge.send("device_info")

    assert bridge.connected
    assert len(fake_ports) == 2
    assert b"device_info" in bytes(fake_ports[1].written)
    bridge.close()


def test_reopening_redetects_the_port(fake_ports, monkeypatch):
    """The device may come back on a different COM number while we were away."""
    bridge = FlipperBridge(idle_timeout=NEVER)
    stale(bridge)
    bridge.release_if_idle()
    monkeypatch.setattr(FlipperBridge, "_auto_detect", staticmethod(lambda: "COM_MOVED"))

    bridge.send("device_info")

    assert bridge.port == "COM_MOVED"
    bridge.close()


def test_reopening_reports_a_device_that_did_not_come_back(fake_ports, monkeypatch):
    from flipper_mcp.bridge import FlipperError

    bridge = FlipperBridge(idle_timeout=NEVER)
    stale(bridge)
    bridge.release_if_idle()

    def gone() -> str:
        raise FlipperError("No Flipper Zero detected on Windows.")

    monkeypatch.setattr(FlipperBridge, "_auto_detect", staticmethod(gone))

    with pytest.raises(FlipperError, match="No Flipper Zero detected"):
        bridge.send("device_info")
    bridge.close()


def test_a_release_and_reopen_cycle_leaves_one_live_reader(fake_ports):
    bridge = FlipperBridge(idle_timeout=NEVER)
    for _ in range(3):
        stale(bridge)
        assert bridge.release_if_idle()
        bridge.send("device_info")

    assert bridge._reader.is_alive()
    alive = [t for t in threading.enumerate() if t.name == "flipper-reader" and t.is_alive()]
    assert len(alive) == 1, f"leaked reader threads: {len(alive)}"
    bridge.close()


# -- never mid-operation ---------------------------------------------------


def test_an_operation_in_flight_is_never_reclaimed(fake_ports):
    """The sweep runs on its own thread, so it can land at any moment."""
    bridge = FlipperBridge(idle_timeout=NEVER)
    stale(bridge)
    verdicts: list[bool] = []

    with bridge.hold():
        # From another thread, exactly as the janitor would arrive.
        sweep = threading.Thread(target=lambda: verdicts.append(bridge.release_if_idle()))
        sweep.start()
        sweep.join(timeout=5.0)
        assert bridge.connected, "the port was pulled out mid-operation"

    assert verdicts == [False]
    bridge.close()


def test_the_sweep_declines_from_inside_its_own_operation(fake_ports):
    """The state lock is reentrant, so same-thread callers need a real guard.

    Without the busy count, a call reaching here from inside an operation
    would take the lock happily and release the port under itself.
    """
    bridge = FlipperBridge(idle_timeout=NEVER)
    stale(bridge)

    with bridge.hold():
        assert bridge.release_if_idle() is False
        assert bridge.connected
    bridge.close()


def test_holding_spans_several_commands(fake_ports):
    """One connection for the whole block — what an RPC session depends on."""
    bridge = FlipperBridge(idle_timeout=NEVER)

    with bridge.hold():
        bridge.send("loader info")
        stale(bridge)
        bridge.send("loader info")

    assert len(fake_ports) == 1, "the port was reopened inside a hold"
    bridge.close()


def test_closing_stops_the_sweep(fake_ports):
    bridge = FlipperBridge(idle_timeout=0.2)
    janitor = bridge._janitor

    bridge.close()

    janitor.join(timeout=5.0)
    assert not janitor.is_alive()
    assert not bridge.connected


# -- sharing the port ------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # macOS / Linux
        "could not open port '/dev/cu.usbmodemflip_X': [Errno 16] Resource busy",
        # Windows
        "could not open port 'COM10': PermissionError(13, 'Access is denied.', None, 5)",
    ],
)
def test_a_port_held_elsewhere_is_reported_in_plain_english(
    monkeypatch, message
):
    """Parking the port makes contention routine, so the message must land.

    Both platforms' wordings have to be recognised: an earlier version matched
    only the POSIX one, leaving Windows users reading a raw ctypes error.
    """
    import serial as pyserial

    from flipper_mcp import bridge as bridge_module
    from flipper_mcp.bridge import FlipperError

    def busy(port, **kwargs):
        raise pyserial.SerialException(message)

    monkeypatch.setattr(bridge_module.serial, "Serial", busy)
    monkeypatch.setattr(FlipperBridge, "_auto_detect", staticmethod(lambda: "COM_TEST"))

    with pytest.raises(FlipperError, match="busy — another app has it open"):
        FlipperBridge()
