from dataclasses import dataclass
import json
import struct

import numpy as np


MESSAGE_VERSION = 1
MSG_TYPE_FRAME = 1
MSG_TYPE_VOICE_BURST = 2
MSG_TYPE_STATUS = 3

FRAME_FLAG_SYNC_DETECTED = 1 << 0
FRAME_FLAG_CLIPPED = 1 << 1
FRAME_FLAG_CONFIDENCE_LOW = 1 << 2

VOICE_FORMAT_RAW_3600 = 1
VOICE_FORMAT_AMBE_2450 = 2

VOICE_BURST_BLOCK_BYTES_RAW_3600 = 9
VOICE_BURST_BLOCK_BYTES_AMBE_2450 = 7

STATUS_SOURCE_PROTOCOL = 1
STATUS_SOURCE_AUDIO = 2
STATUS_SOURCE_LAUNCHER = 3

FRAME_HEADER = struct.Struct("<HHIQHHHH")
VOICE_HEADER = struct.Struct("<HHIHBHIBB")
STATUS_HEADER = struct.Struct("<HHIQHHI")


def _voice_payload_block_size(payload_format: int) -> int:
    if payload_format == VOICE_FORMAT_RAW_3600:
        return VOICE_BURST_BLOCK_BYTES_RAW_3600
    if payload_format == VOICE_FORMAT_AMBE_2450:
        return VOICE_BURST_BLOCK_BYTES_AMBE_2450
    raise ValueError(f"Unsupported voice payload format: {payload_format}")


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    monotonic_ns: int
    channel_id: int
    flags: int
    symbols: np.ndarray

    def encode(self) -> bytes:
        symbols = np.asarray(self.symbols, dtype=np.int8)
        return FRAME_HEADER.pack(
            MESSAGE_VERSION,
            MSG_TYPE_FRAME,
            self.sequence,
            self.monotonic_ns,
            self.channel_id,
            self.flags,
            symbols.size,
            0,
        ) + symbols.tobytes()

    @classmethod
    def decode(cls, payload: bytes):
        if len(payload) < FRAME_HEADER.size:
            raise ValueError("Frame packet is shorter than the header size.")

        version, msg_type, sequence, monotonic_ns, channel_id, flags, symbol_count, _ = FRAME_HEADER.unpack(
            payload[: FRAME_HEADER.size]
        )
        if version != MESSAGE_VERSION:
            raise ValueError(f"Unsupported message version: {version}")
        if msg_type != MSG_TYPE_FRAME:
            raise ValueError(f"Unsupported message type: {msg_type}")

        expected_size = FRAME_HEADER.size + symbol_count
        if len(payload) != expected_size:
            raise ValueError(
                f"Frame packet size mismatch: expected {expected_size} bytes, got {len(payload)} bytes."
            )

        symbols = np.frombuffer(payload[FRAME_HEADER.size:], dtype=np.int8, count=symbol_count).copy()
        return cls(
            sequence=sequence,
            monotonic_ns=monotonic_ns,
            channel_id=channel_id,
            flags=flags,
            symbols=symbols,
        )


@dataclass(frozen=True)
class VoiceBurstPacket:
    sequence: int
    channel_id: int
    call_stat: int
    key_id: int
    burst_index: int
    payload_format: int
    payload: bytes

    def encode(self) -> bytes:
        block_size = _voice_payload_block_size(self.payload_format)
        if len(self.payload) % block_size != 0:
            raise ValueError(
                f"Voice burst payload size {len(self.payload)} is not a multiple of {block_size} bytes."
            )

        block_count = len(self.payload) // block_size
        if block_count > 0xFF:
            raise ValueError("Voice burst block count exceeds the header limit.")

        return VOICE_HEADER.pack(
            MESSAGE_VERSION,
            MSG_TYPE_VOICE_BURST,
            self.sequence,
            self.channel_id,
            self.call_stat,
            self.key_id,
            self.burst_index,
            block_count,
            self.payload_format,
        ) + self.payload

    @classmethod
    def decode(cls, payload: bytes):
        if len(payload) < VOICE_HEADER.size:
            raise ValueError("Voice burst packet is shorter than the header size.")

        version, msg_type, sequence, channel_id, call_stat, key_id, burst_index, block_count, payload_format = VOICE_HEADER.unpack(
            payload[: VOICE_HEADER.size]
        )
        if version != MESSAGE_VERSION:
            raise ValueError(f"Unsupported message version: {version}")
        if msg_type != MSG_TYPE_VOICE_BURST:
            raise ValueError(f"Unsupported message type: {msg_type}")

        burst_payload = payload[VOICE_HEADER.size:]
        block_size = _voice_payload_block_size(payload_format)
        if block_count * block_size != len(burst_payload):
            raise ValueError("Voice burst payload size mismatch.")

        return cls(
            sequence=sequence,
            channel_id=channel_id,
            call_stat=call_stat,
            key_id=key_id,
            burst_index=burst_index,
            payload_format=payload_format,
            payload=burst_payload,
        )


@dataclass(frozen=True)
class StatusPacket:
    sequence: int
    monotonic_ns: int
    source: int
    channel_id: int
    payload: bytes

    def encode(self) -> bytes:
        return STATUS_HEADER.pack(
            MESSAGE_VERSION,
            MSG_TYPE_STATUS,
            self.sequence,
            self.monotonic_ns,
            self.source,
            self.channel_id,
            len(self.payload),
        ) + self.payload

    def to_dict(self):
        return json.loads(self.payload.decode("utf-8"))

    @classmethod
    def from_dict(cls, sequence: int, monotonic_ns: int, source: int, channel_id: int, payload_dict: dict):
        payload = json.dumps(payload_dict, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return cls(
            sequence=sequence,
            monotonic_ns=monotonic_ns,
            source=source,
            channel_id=channel_id,
            payload=payload,
        )

    @classmethod
    def decode(cls, payload: bytes):
        if len(payload) < STATUS_HEADER.size:
            raise ValueError("Status packet is shorter than the header size.")

        version, msg_type, sequence, monotonic_ns, source, channel_id, payload_length = STATUS_HEADER.unpack(
            payload[: STATUS_HEADER.size]
        )
        if version != MESSAGE_VERSION:
            raise ValueError(f"Unsupported message version: {version}")
        if msg_type != MSG_TYPE_STATUS:
            raise ValueError(f"Unsupported message type: {msg_type}")

        status_payload = payload[STATUS_HEADER.size:]
        if len(status_payload) != payload_length:
            raise ValueError("Status payload size mismatch.")

        return cls(
            sequence=sequence,
            monotonic_ns=monotonic_ns,
            source=source,
            channel_id=channel_id,
            payload=status_payload,
        )
