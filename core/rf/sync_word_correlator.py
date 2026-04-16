import time

import numpy as np
from gnuradio import gr

from core.protocol.frame_layout import quantize_float_symbols
from ipc.message_schema import FRAME_FLAG_SYNC_DETECTED, FramePacket
from ipc.transport.uds_seqpacket import UdsSeqpacketServer, resolve_frame_socket_path


class SyncWordCorrelator(gr.sync_block):
    def __init__(
        self,
        channel_count,
        sync_word=None,
        packet_len=192,
        socket_path=None,
        abs_limit=None,
        threshold_ratio=None,
        error_threshold_ratio=None,
    ):
        self.channel_count = int(channel_count)
        gr.sync_block.__init__(
            self,
            name="Sync Word Correlator",
            in_sig=[np.float32] * self.channel_count,
            out_sig=None,
        )

        if threshold_ratio is None and error_threshold_ratio is None:
            raise ValueError("Either threshold_ratio or error_threshold_ratio must be provided.")
        if threshold_ratio is not None and error_threshold_ratio is not None:
            raise ValueError("threshold_ratio and error_threshold_ratio are mutually exclusive.")

        if sync_word is None:
            sync_word = [-3.0, +1.0, -3.0, +3.0, -3.0, -3.0, +3.0, +3.0, -1.0, +3.0]
        self.sync_word = np.asarray(sync_word, dtype=np.float32)
        self.sync_word_length = len(self.sync_word)

        self.packet_len = int(packet_len)
        if self.packet_len < self.sync_word_length:
            raise ValueError("packet_len must be >= length of sync_word")

        sync_word_energy = float(np.dot(self.sync_word, self.sync_word))
        if threshold_ratio is not None:
            self.detection_mode = "correlation"
            self.match_threshold = sync_word_energy * float(threshold_ratio)
            self.abs_limit = float(abs_limit if abs_limit is not None else 4.0)
        else:
            self.detection_mode = "sse"
            self.match_threshold = sync_word_energy * float(error_threshold_ratio)
            self.abs_limit = None

        self.shift_registers = [np.zeros(self.sync_word_length, dtype=np.float32) for _ in range(self.channel_count)]
        self.collecting = [False] * self.channel_count
        self.packet_bufs = [None] * self.channel_count
        self.samples_collected = [0] * self.channel_count
        self.sequence = 0

        resolved_socket_path = resolve_frame_socket_path(channel_count=self.channel_count, socket_path=socket_path)
        self.transport = UdsSeqpacketServer(resolved_socket_path)

    def stop(self):
        self.transport.close()
        return True

    def _start_collection(self, channel_index, initial_window):
        self.packet_bufs[channel_index] = np.empty(self.packet_len, dtype=np.float32)
        self.packet_bufs[channel_index][: self.sync_word_length] = initial_window
        self.samples_collected[channel_index] = self.sync_word_length
        self.collecting[channel_index] = True

    def _sync_detected(self, channel_index):
        if self.detection_mode == "correlation":
            correlation = float(np.dot(self.shift_registers[channel_index], self.sync_word))
            return correlation >= self.match_threshold

        squared_error = float(np.sum((self.shift_registers[channel_index] - self.sync_word) ** 2))
        return squared_error <= self.match_threshold

    def _append_symbol(self, channel_index, symbol):
        if not self.collecting[channel_index]:
            return

        collected = self.samples_collected[channel_index]
        if collected < self.packet_len:
            self.packet_bufs[channel_index][collected] = float(symbol)
            self.samples_collected[channel_index] += 1

        if self.samples_collected[channel_index] < self.packet_len:
            return

        packet = FramePacket(
            sequence=self.sequence,
            monotonic_ns=time.monotonic_ns(),
            channel_id=channel_index + 1,
            flags=FRAME_FLAG_SYNC_DETECTED,
            symbols=quantize_float_symbols(self.packet_bufs[channel_index]),
        )
        self.sequence += 1
        self.transport.send(packet.encode())

        self.collecting[channel_index] = False
        self.packet_bufs[channel_index] = None
        self.samples_collected[channel_index] = 0

    def work(self, input_items, output_items):
        sample_count = len(input_items[0])

        for channel_index in range(self.channel_count):
            in_data = input_items[channel_index]
            for sample_index in range(sample_count):
                symbol = float(in_data[sample_index])
                if self.abs_limit is not None:
                    if symbol > self.abs_limit:
                        symbol = self.abs_limit
                    elif symbol < -self.abs_limit:
                        symbol = -self.abs_limit

                if self.collecting[channel_index]:
                    self._append_symbol(channel_index, symbol)
                    continue

                self.shift_registers[channel_index][:-1] = self.shift_registers[channel_index][1:]
                self.shift_registers[channel_index][-1] = symbol
                if self._sync_detected(channel_index):
                    self._start_collection(channel_index, self.shift_registers[channel_index].copy())

        return sample_count