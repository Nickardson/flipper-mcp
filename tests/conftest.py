"""Shared fixtures for the bridge's connection-lifecycle tests.

The bridge opens, drops, and reopens the serial port on its own — on write
failure, on a dead reader thread, and after an idle period. Exercising that
against real hardware would mean physically replugging a Flipper, so these
tests swap ``serial.Serial`` for a fake whose failure modes are settable.
"""

from __future__ import annotations

import time

import pytest
import serial

from flipper_mcp import bridge as bridge_module
from flipper_mcp.bridge import FlipperBridge


class FakeSerial:
    """A serial port that can be made to fail the way a stale handle does.

    ``failing`` models a handle that is stale from the outset. ``fail_after``
    models one that goes stale partway through an operation: that many writes
    succeed and every later one fails. Each instance is one *handle*, so a
    reconnect produces a fresh one — exactly the state change the bridge is
    supposed to bring about, and what the tests assert on.
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
        if self.failing or (
            self.fail_after is not None and self._writes > self.fail_after
        ):
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
        # instantly would turn every reader thread into a hot spin.
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
    """Install FakeSerial and record every handle the bridge opens.

    The list is the assertion surface: its length counts reconnects, and each
    entry exposes what that particular handle saw.
    """
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
