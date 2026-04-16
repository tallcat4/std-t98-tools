THUMBDV_MAP = [
    0, 18, 36, 1, 19, 37, 2, 20, 38,
    3, 21, 39, 4, 22, 40, 5, 23, 41,
    6, 24, 42, 7, 25, 43, 8, 26, 44,
    9, 27, 45, 10, 28, 46, 11, 29, 47,
    12, 30, 48, 13, 31, 14, 32, 15, 33,
    16, 34, 17, 35,
]


def descramble_burst(payloads, key_196):
    result = []
    for frame_index in range(4):
        payload = payloads[frame_index]
        key_slice = key_196[frame_index * 49 : (frame_index + 1) * 49]

        raw_bits = []
        for byte_val in payload:
            for bit_pos in range(7, -1, -1):
                raw_bits.append((byte_val >> bit_pos) & 1)
        raw_bits = raw_bits[:49]

        descrambled_bits = [raw_bits[index] ^ key_slice[THUMBDV_MAP[index]] for index in range(49)]

        value = 0
        for bit in descrambled_bits:
            value = (value << 1) | bit
        value <<= 7
        result.append(value.to_bytes(7, byteorder="big"))
    return result