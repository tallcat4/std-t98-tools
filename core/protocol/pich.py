from core.protocol.sacch import TRELLIS


def decode_pich(pich_bits_str: str) -> dict:
    if len(pich_bits_str) != 144:
        raise ValueError(f"PICH must be exactly 144 bits, got {len(pich_bits_str)}")

    pich_bits = [int(bit) for bit in pich_bits_str]

    deinterleaved = [0] * 144
    for col in range(16):
        for row in range(9):
            deinterleaved[row * 16 + col] = pich_bits[col * 9 + row]

    depunctured = []
    index = 0
    for _ in range(48):
        depunctured.append(deinterleaved[index])
        index += 1
        depunctured.append(None)
        depunctured.append(deinterleaved[index])
        index += 1
        depunctured.append(deinterleaved[index])
        index += 1

    metrics = {state: (float("inf"), []) for state in range(16)}
    metrics[0] = (0, [])

    for step in range(96):
        r1 = depunctured[step * 2]
        r2 = depunctured[step * 2 + 1]
        new_metrics = {state: (float("inf"), []) for state in range(16)}
        valid_bits = (0, 1) if step < 92 else (0,)
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

    msg_bits = decoded_bits[:80]
    crc_bits_received = decoded_bits[80:92]

    register = [1] * 12
    for bit in msg_bits:
        feedback = bit ^ register[11]
        next_register = [0] * 12
        next_register[0] = feedback
        next_register[1] = register[0] ^ feedback
        next_register[2] = register[1] ^ feedback
        next_register[3] = register[2] ^ feedback
        next_register[4] = register[3]
        next_register[5] = register[4]
        next_register[6] = register[5]
        next_register[7] = register[6]
        next_register[8] = register[7]
        next_register[9] = register[8]
        next_register[10] = register[9]
        next_register[11] = register[10] ^ feedback
        register = next_register

    crc_calculated = register[::-1]
    crc_ok = crc_calculated == crc_bits_received
    if not crc_ok and register == crc_bits_received:
        crc_ok = True
        crc_calculated = register

    csm_bits = msg_bits[:36]
    csm_str = ""
    for index in range(9):
        digit_bits = csm_bits[index * 4 : index * 4 + 4]
        digit_val = int("".join(map(str, digit_bits)), 2)
        csm_str += f"{digit_val:X}"

    reserve_bits = msg_bits[36:80]

    return {
        "CSM": csm_str,
        "Reserve": "".join(map(str, reserve_bits)),
        "CRC_OK": crc_ok,
        "CRC_Recv": "".join(map(str, crc_bits_received)),
        "CRC_Calc": "".join(map(str, crc_calculated)),
        "BitErrors": corrected_errors,
    }