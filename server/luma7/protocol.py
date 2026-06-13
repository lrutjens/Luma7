"""Binary upload protocol and payload parsing."""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass
class QueryPayload:
    jpeg: bytes
    wav: bytes

    @property
    def has_image(self) -> bool:
        return len(self.jpeg) > 0


def parse_query_body(body: bytes) -> QueryPayload:
    if len(body) < 4:
        raise ValueError("payload too short")
    jpeg_len = struct.unpack(">I", body[:4])[0]
    if jpeg_len > len(body) - 4:
        raise ValueError("invalid jpeg length")
    jpeg = body[4 : 4 + jpeg_len]
    wav = body[4 + jpeg_len :]
    if not wav:
        raise ValueError("missing wav audio")
    return QueryPayload(jpeg=jpeg, wav=wav)
