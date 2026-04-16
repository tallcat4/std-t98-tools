def split_traffic_blocks(fields):
    return [
        fields["TCH1"][:72],
        fields["TCH1"][72:],
        fields["TCH2"][:72],
        fields["TCH2"][72:],
    ]


def block_strings_to_payloads_3600(blocks):
    return [int(block, 2).to_bytes(9, byteorder="big") for block in blocks]


def append_blocks_to_ambe_file(file_path, blocks):
    with open(file_path, "ab") as handle:
        for payload in block_strings_to_payloads_3600(blocks):
            handle.write(b"\x48" + payload)


def append_blocks_to_burst_file(file_path, blocks):
    with open(file_path, "ab") as handle:
        handle.write(b"\xFF")
        for payload in block_strings_to_payloads_3600(blocks):
            handle.write(b"\x48" + payload)