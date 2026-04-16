import numpy as np


def dewhiten(symbol_seq):
    if len(symbol_seq) < 192:
        return symbol_seq

    lfsr = [0, 1, 1, 1, 0, 0, 1, 0, 0]
    whitening_seq = []
    for _ in range(182):
        s0 = lfsr[-1]
        s4 = lfsr[4]
        new_s8 = s4 ^ s0
        whitening_seq.append(s0)
        lfsr = [new_s8] + lfsr[:-1]

    signs = np.array([1 if bit == 0 else -1 for bit in whitening_seq], dtype=np.int8)
    output_seq = np.asarray(symbol_seq, dtype=np.int8).copy()
    output_seq[10:] = output_seq[10:] * signs
    return output_seq