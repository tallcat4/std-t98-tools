import pytest
import numpy as np

from ipc.message_schema import (
    FRAME_FLAG_SYNC_DETECTED,
    FramePacket,
    STATUS_SOURCE_PROTOCOL,
    StatusPacket,
    VOICE_FORMAT_AMBE_2450,
    VOICE_FORMAT_RAW_3600,
    VoiceBurstPacket,
)


def test_frame_packet_roundtrip():
    symbols = np.array(([-3, -1, 1, 3] * 48)[:192], dtype=np.int8)
    packet = FramePacket(
        sequence=17,
        monotonic_ns=123456789,
        channel_id=4,
        flags=FRAME_FLAG_SYNC_DETECTED,
        symbols=symbols,
    )

    decoded = FramePacket.decode(packet.encode())

    assert decoded.sequence == 17
    assert decoded.monotonic_ns == 123456789
    assert decoded.channel_id == 4
    assert decoded.flags == FRAME_FLAG_SYNC_DETECTED
    assert np.array_equal(decoded.symbols, symbols)


def test_voice_burst_packet_roundtrip():
    payload = bytes(range(36))
    packet = VoiceBurstPacket(
        sequence=8,
        channel_id=2,
        call_stat=1,
        key_id=3,
        burst_index=99,
        payload_format=VOICE_FORMAT_RAW_3600,
        payload=payload,
    )

    decoded = VoiceBurstPacket.decode(packet.encode())

    assert decoded.sequence == 8
    assert decoded.channel_id == 2
    assert decoded.call_stat == 1
    assert decoded.key_id == 3
    assert decoded.burst_index == 99
    assert decoded.payload_format == VOICE_FORMAT_RAW_3600
    assert decoded.payload == payload


def test_voice_burst_packet_roundtrip_for_ambe_2450():
    payload = bytes(range(28))
    packet = VoiceBurstPacket(
        sequence=9,
        channel_id=7,
        call_stat=2,
        key_id=11,
        burst_index=101,
        payload_format=VOICE_FORMAT_AMBE_2450,
        payload=payload,
    )

    decoded = VoiceBurstPacket.decode(packet.encode())

    assert decoded.sequence == 9
    assert decoded.channel_id == 7
    assert decoded.call_stat == 2
    assert decoded.key_id == 11
    assert decoded.burst_index == 101
    assert decoded.payload_format == VOICE_FORMAT_AMBE_2450
    assert decoded.payload == payload


def test_voice_burst_packet_rejects_invalid_raw_3600_payload_length():
    packet = VoiceBurstPacket(
        sequence=10,
        channel_id=1,
        call_stat=0,
        key_id=0,
        burst_index=0,
        payload_format=VOICE_FORMAT_RAW_3600,
        payload=b"\x00" * 35,
    )

    with pytest.raises(ValueError, match="multiple of 9 bytes"):
        packet.encode()


def test_status_packet_roundtrip():
    packet = StatusPacket.from_dict(
        sequence=12,
        monotonic_ns=987654321,
        source=STATUS_SOURCE_PROTOCOL,
        channel_id=2,
        payload_dict={
            "event": "channel_state",
            "rx_status": "OPEN",
            "protocol_status": "Sync Burst",
        },
    )

    decoded = StatusPacket.decode(packet.encode())

    assert decoded.sequence == 12
    assert decoded.monotonic_ns == 987654321
    assert decoded.source == STATUS_SOURCE_PROTOCOL
    assert decoded.channel_id == 2
    assert decoded.to_dict() == {
        "event": "channel_state",
        "rx_status": "OPEN",
        "protocol_status": "Sync Burst",
    }