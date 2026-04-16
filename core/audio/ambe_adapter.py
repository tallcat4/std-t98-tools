import importlib

import numpy as np


def _pyambelib_module():
    return importlib.import_module("pyambelib")


def create_decoder():
    return _pyambelib_module().AmbeDecoder()


def bits_to_bytes(bits):
    result = []
    for index in range(0, len(bits), 8):
        byte_val = 0
        chunk_len = min(8, len(bits) - index)
        for bit_offset in range(chunk_len):
            byte_val = (byte_val << 1) | bits[index + bit_offset]
        if chunk_len < 8:
            byte_val <<= 8 - chunk_len
        result.append(byte_val)
    return result


def fec_demod_to_2450_payload(payload_3600: bytes) -> bytes:
    try:
        bits = _pyambelib_module().fec_demod(payload_3600)
    except Exception:
        return b"\x00" * 7

    bytes_from_bits = bits_to_bytes(bits)
    if len(bytes_from_bits) >= 7:
        return bytes(bytes_from_bits[:7])
    return b"\x00" * 7


def payloads_3600_to_2450(payloads_3600):
    return [fec_demod_to_2450_payload(payload) for payload in payloads_3600]


def decode_2450_payloads_to_pcm(payloads_2450, decoder, upsample_factor=6):
    pcm_out = b""
    for payload in payloads_2450:
        samples = decoder.decode_2450(payload)
        if not samples:
            continue
        audio_array_8k = np.array(samples, dtype=np.int16)
        audio_array_48k = np.repeat(audio_array_8k, upsample_factor)
        pcm_out += audio_array_48k.tobytes()
    return pcm_out