import numpy as np


FRAME_STRUCTURE = (
    ("SW", 20),
    ("RICH", 16),
    ("SACCH", 60),
    ("TCH1", 144),
    ("TCH2", 144),
)


def quantize_float_symbols(raw_symbols):
    raw_array = np.asarray(raw_symbols, dtype=np.float32)
    symbols = np.zeros(raw_array.shape, dtype=np.int8)
    symbols[raw_array > 2.0] = 3
    symbols[(raw_array > 0.0) & (raw_array <= 2.0)] = 1
    symbols[(raw_array > -2.0) & (raw_array <= 0.0)] = -1
    symbols[raw_array <= -2.0] = -3
    return symbols


def symbols_to_bit_values(symbols):
    symbol_array = np.asarray(symbols, dtype=np.int8)
    mapped = np.zeros(symbol_array.shape, dtype=np.uint8)
    mapped[symbol_array == -3] = 3
    mapped[symbol_array == -1] = 2
    mapped[symbol_array == 1] = 0
    mapped[symbol_array == 3] = 1
    return mapped


def symbols_to_binary_string(symbols):
    mapped = symbols_to_bit_values(symbols)
    return "".join(f"{value:02b}" for value in mapped)


def split_frame_binary_string(frame_bits):
    fields = {}
    current_bit_index = 0
    for name, bit_length in FRAME_STRUCTURE:
        fields[name] = frame_bits[current_bit_index : current_bit_index + bit_length]
        current_bit_index += bit_length
    return fields


def parse_frame_symbols(symbols):
    return split_frame_binary_string(symbols_to_binary_string(symbols))


def format_binary_string(bin_str, chunk_size=8):
    return " ".join(
        bin_str[index : index + chunk_size]
        for index in range(0, len(bin_str), chunk_size)
    )