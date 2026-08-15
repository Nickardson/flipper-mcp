"""Framebuffer capture: Flipper screen -> PNG plus a verification envelope.

Framebuffer layout (confirmed against lab.flipper.net's canvas renderer,
frontend/src/shared/lib/flipperJs/frameRenderer.ts): 1024 bytes covering
128x64 pixels, page-major. Each byte holds 8 *vertical* pixels:

    byte  = data[(y >> 3) * 128 + x]
    pixel = byte & (1 << (y & 7))        set bit = ink

The status bar carries the clock, battery level, Bluetooth state and the SD
icon, all of which change on their own schedule regardless of what app is in
front. Hashing the whole frame therefore produces an identity that goes stale
within a minute, so ``body_sha256`` covers only the rows beneath it and the
volatile whole-frame digest is reported separately as ``frame_sha256``.

The crop is row-exact rather than page-aligned. An earlier version excluded
one 8-row page because that is a clean byte slice, but a desktop capture
shows the bar running past row 8 — the bottom of the clock digits survived
into the "stable" digest, which would have changed every minute.

Note that neither digest can identify a screen that animates. Momentum
marquee-scrolls the label of the *selected* item in its app grid, and its
icons animate, so those frames differ every capture. Digests are for static
screens; for anything animated the image is the reliable signal.

PNG encoding is hand-rolled on stdlib zlib. An 8-bit grayscale image at 4x
nearest-neighbour costs roughly 175 vision tokens, which is *cheaper* than
rendering the same frame as braille art and far more legible.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from typing import Optional

from .bridge import FlipperBridge, FlipperError
from .rpc import FIELD_GUI_START_SCREEN_STREAM, FIELD_GUI_STOP_SCREEN_STREAM, rpc_session

WIDTH = 128
HEIGHT = 64
FRAME_BYTES = WIDTH * HEIGHT // 8

# Pixel rows occupied by the status bar, excluded from ``body_sha256``.
# Taken from the firmware's canvas status-bar height; a desktop capture puts
# the bar's lower edge in this neighbourhood. Erring one row large is the safe
# direction — it drops a row of app content rather than letting the clock in.
STATUS_BAR_ROWS = 13

INK = 0x00  # black
PAPER = 0xFF  # white — max contrast beats reproducing the orange backlight

ORIENTATIONS = {0: "horizontal", 1: "horizontal_flip", 2: "vertical", 3: "vertical_flip"}


def unpack(data: bytes, orientation: int = 0) -> list[list[bool]]:
    """Expand the packed framebuffer into [y][x] booleans, ink = True."""
    if len(data) < FRAME_BYTES:
        raise FlipperError(
            f"Short framebuffer: got {len(data)} bytes, expected {FRAME_BYTES}."
        )
    rows = [
        [bool(data[(y >> 3) * WIDTH + x] & (1 << (y & 7))) for x in range(WIDTH)]
        for y in range(HEIGHT)
    ]
    if orientation == 1:  # 180 degrees; matches the web app's only rotation case
        rows = [list(reversed(row)) for row in reversed(rows)]
    return rows


def pack_rows(rows: list[list[bool]]) -> bytes:
    """Pack [y][x] booleans row-major, MSB first — the basis for the digests.

    Hashing the displayed pixels rather than the raw transport bytes lets the
    crop land on an arbitrary row, and keeps a digest meaningful across the
    180-degree orientation the frame may arrive in.
    """
    out = bytearray()
    for row in rows:
        accumulator = 0
        for bit, pixel in enumerate(row):
            accumulator = (accumulator << 1) | pixel
            if bit % 8 == 7:
                out.append(accumulator)
                accumulator = 0
    return bytes(out)


def to_png(rows: list[list[bool]], scale: int = 4) -> bytes:
    """Encode [y][x] booleans as an 8-bit grayscale PNG, nearest-neighbour."""
    width, height = WIDTH * scale, HEIGHT * scale
    raw = bytearray()
    for row in rows:
        line = bytearray()
        for pixel in row:
            line.extend(bytes([INK if pixel else PAPER]) * scale)
        for _ in range(scale):
            raw.append(0)  # filter type 0 (None)
            raw.extend(line)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
            chunk(b"IEND", b""),
        ]
    )


def _foreground_app(bridge: FlipperBridge) -> Optional[str]:
    """Best-effort app name from `loader info`; never fails the capture."""
    try:
        output = bridge.send("loader info", timeout=3.0).strip()
    except Exception:
        return None
    for line in output.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            if key.strip().lower() in {"app", "name", "running"}:
                return value.strip()
    return output.splitlines()[0].strip() if output else None


def capture(bridge: FlipperBridge, scale: int = 4, timeout: float = 3.0) -> dict:
    """Grab one frame and return {png, envelope} ready for the MCP layer."""
    if not 1 <= scale <= 8:
        raise FlipperError(f"scale must be between 1 and 8, got {scale}")

    # Read the app name *before* entering RPC — the text CLI is unreachable
    # for the duration of the session.
    app = _foreground_app(bridge)

    with rpc_session(bridge, handshake_timeout=timeout) as channel:
        channel.send(FIELD_GUI_START_SCREEN_STREAM)
        try:
            data, orientation = channel.await_frame(timeout=timeout)
        finally:
            channel.send(FIELD_GUI_STOP_SCREEN_STREAM)

    rows = unpack(data, orientation)
    return {
        "png": to_png(rows, scale=scale),
        "envelope": {
            "app": app,
            "width": WIDTH,
            "height": HEIGHT,
            "scale": scale,
            "orientation": ORIENTATIONS.get(orientation, str(orientation)),
            "body_sha256": hashlib.sha256(pack_rows(rows[STATUS_BAR_ROWS:])).hexdigest(),
            "frame_sha256": hashlib.sha256(pack_rows(rows)).hexdigest(),
            "text": None,  # reserved: bitmap-font OCR
        },
    }
