def generate_pn_sequence_196(initial_state):
    state = initial_state
    pn_bits = []
    for _ in range(196):
        output = state & 1
        pn_bits.append(output)
        s0 = state & 1
        s1 = (state >> 1) & 1
        feedback = s0 ^ s1
        state = state >> 1
        if feedback:
            state |= 1 << 14
    return pn_bits