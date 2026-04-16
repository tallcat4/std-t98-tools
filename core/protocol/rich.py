def decode_rich(rich_bits_str: str) -> dict:
    if len(rich_bits_str) != 16:
        raise ValueError(f"RICH must be exactly 16 bits, got {len(rich_bits_str)} bits.")

    rich_decoded_bits = []
    for index in range(0, len(rich_bits_str), 2):
        pair = rich_bits_str[index : index + 2]
        if pair in ("00", "01"):
            rich_decoded_bits.append(0)
        elif pair in ("10", "11"):
            rich_decoded_bits.append(1)
        else:
            raise ValueError(f"Invalid 2-bit pair '{pair}' in RICH bits.")

    val_f = rich_decoded_bits[0]
    val_res1 = rich_decoded_bits[1]
    val_res2 = rich_decoded_bits[2]
    val_m = (rich_decoded_bits[3] << 2) | (rich_decoded_bits[4] << 1) | rich_decoded_bits[5]
    val_d = rich_decoded_bits[6]
    val_p = rich_decoded_bits[7]

    calc_parity = sum(rich_decoded_bits[:7]) % 2
    parity_ok = calc_parity == val_p

    return {
        "F": val_f,
        "Res1": val_res1,
        "Res2": val_res2,
        "M": val_m,
        "D": val_d,
        "Parity": val_p,
        "Parity_OK": parity_ok,
    }