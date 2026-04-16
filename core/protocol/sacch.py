def _build_trellis():
    trellis = {}
    for state in range(16):
        for bit in (0, 1):
            x_t = bit
            x_t1 = (state >> 3) & 1
            x_t2 = (state >> 2) & 1
            x_t3 = (state >> 1) & 1
            x_t4 = state & 1
            next_state = (bit << 3) | (state >> 1)
            g1_out = x_t ^ x_t3 ^ x_t4
            g2_out = x_t ^ x_t1 ^ x_t2 ^ x_t4
            trellis[(state, bit)] = (next_state, g1_out, g2_out)
    return trellis


TRELLIS = _build_trellis()


def _bits_to_int(bit_list):
    return int("".join(map(str, bit_list)), 2)


def decode_sacch(sacch_bits_str: str) -> dict:
    if len(sacch_bits_str) != 60:
        raise ValueError(f"SACCH must be exactly 60 bits, got {len(sacch_bits_str)} bits.")

    sacch_bits = [int(bit) for bit in sacch_bits_str]

    deinterleaved = [0] * 60
    for col in range(12):
        for row in range(5):
            deinterleaved[row * 12 + col] = sacch_bits[col * 5 + row]

    depunctured = []
    index = 0
    for _ in range(6):
        for puncture_index in range(6):
            depunctured.append(deinterleaved[index])
            index += 1
            if puncture_index in (2, 5):
                depunctured.append(None)
            else:
                depunctured.append(deinterleaved[index])
                index += 1

    metrics = {state: (float("inf"), []) for state in range(16)}
    metrics[0] = (0, [])

    for step in range(36):
        r1 = depunctured[step * 2]
        r2 = depunctured[step * 2 + 1]
        new_metrics = {state: (float("inf"), []) for state in range(16)}
        valid_bits = (0, 1) if step < 32 else (0,)
        for state, (cost, history) in metrics.items():
            if cost == float("inf"):
                continue
            for bit in valid_bits:
                next_state, g1_out, g2_out = TRELLIS[(state, bit)]
                dist = 0
                if r1 is not None and r1 != g1_out:
                    dist += 1
                if r2 is not None and r2 != g2_out:
                    dist += 1
                new_cost = cost + dist
                if new_cost < new_metrics[next_state][0]:
                    new_metrics[next_state] = (new_cost, history + [bit])
        metrics = new_metrics

    decoded_bits = metrics[0][1]
    corrected_errors = metrics[0][0]

    msg_bits = decoded_bits[:26]
    crc_bits_received = decoded_bits[26:32]

    register = [1, 1, 1, 1, 1, 1]
    for bit in msg_bits:
        feedback = bit ^ register[5]
        register = [feedback, register[0] ^ feedback, register[1] ^ feedback, register[2], register[3], register[4] ^ feedback]

    crc_calculated = [register[5], register[4], register[3], register[2], register[1], register[0]]
    crc_ok = crc_calculated == crc_bits_received
    if not crc_ok:
        crc_calculated_rev = [register[0], register[1], register[2], register[3], register[4], register[5]]
        if crc_calculated_rev == crc_bits_received:
            crc_ok = True
            crc_calculated = crc_calculated_rev

    return {
        "F": msg_bits[0],
        "Wr": _bits_to_int(msg_bits[1:3]),
        "MsgType": _bits_to_int(msg_bits[3:8]),
        "CallStat": _bits_to_int(msg_bits[8:10]),
        "UserCode": _bits_to_int(msg_bits[10:19]),
        "MakerCode": _bits_to_int(msg_bits[19:26]),
        "CRC_OK": crc_ok,
        "CRC_Recv": "".join(map(str, crc_bits_received)),
        "CRC_Calc": "".join(map(str, crc_calculated)),
        "BitErrors": corrected_errors,
    }