"""
Strip telnet negotiation bytes from user input lines and parse NAWS for dimensions.

NAWS subnegotiation includes 0x00 in width/height; if mixed into readline() before
CRLF, those bytes must not be sent to the backend as part of username/password.
"""

from __future__ import annotations

from typing import Callable, Optional


# Telnet
IAC = 255
SB = 250
SE = 240
WILL = 251
WONT = 252
DO = 253
DONT = 254
NAWS = 31


def strip_telnet_from_buffer(
    buf: bytes,
    on_naws: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """
    Remove IAC sequences from a single line buffer. NUL bytes are dropped (DB-safe).
    If on_naws is provided, call on_naws(width, height) when NAWS is parsed.
    """
    out = bytearray()
    i = 0
    n = len(buf)

    def _emit_naws(payload: bytes) -> None:
        if on_naws is None or len(payload) < 4:
            return
        w = (payload[0] << 8) | payload[1]
        h = (payload[2] << 8) | payload[3]
        if 20 <= w <= 512:
            on_naws(w, h)

    while i < n:
        c = buf[i]
        if c != IAC:
            if c != 0:
                out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            break
        cmd = buf[i + 1]
        if cmd == IAC:
            out.append(IAC)
            i += 2
            continue
        if cmd in (WILL, WONT, DO, DONT):
            i += 3
            continue
        if cmd == SB:
            if i + 2 >= n:
                break
            opt = buf[i + 2]
            j = i + 3
            while j < n:
                if buf[j] == IAC and j + 1 < n and buf[j + 1] == SE:
                    payload = buf[i + 3 : j]
                    if opt == NAWS:
                        _emit_naws(payload)
                    i = j + 2
                    break
                j += 1
            else:
                i = n
            continue
        i += 2

    return bytes(out)


async def discard_telnet_command_from_reader(reader) -> None:
    """
    After reading IAC (255) from stream, consume the rest of that telnet command.
    Used when reading password one byte at a time.
    """
    cmd_b = await reader.read(1)
    if not cmd_b:
        return
    cmd = cmd_b[0]
    if cmd == IAC:
        return
    if cmd in (WILL, WONT, DO, DONT):
        await reader.read(1)
        return
    if cmd == SB:
        while True:
            b = await reader.read(1)
            if not b:
                return
            if b[0] != IAC:
                continue
            b2 = await reader.read(1)
            if not b2:
                return
            if b2[0] == SE:
                return
