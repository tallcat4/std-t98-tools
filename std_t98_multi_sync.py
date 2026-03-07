#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gnuradio import gr
import zmq
import struct

class sync_word_correlator(gr.sync_block):
    def __init__(self,
                 num_channels,
                 zmq_ip="127.0.0.1",
                 zmq_port=5555,
                 sync_word=None,
                 error_threshold_ratio=0.2,
                 packet_len=192):
        
        self.num_channels = int(num_channels)

        gr.sync_block.__init__(
            self,
            name="Sync Word Correlator",
            in_sig=[np.float32] * self.num_channels,
            out_sig=None
        )

        if sync_word is None:
            sync_word =[-3.0, +1.0, -3.0, +3.0, -3.0, -3.0, +3.0, +3.0, -1.0, +3.0]
        self.sync_word = np.asarray(sync_word, dtype=np.float32)
        self.sw_len = len(self.sync_word)

        sw_energy = float(np.dot(self.sync_word, self.sync_word))
        
        self.error_threshold = sw_energy * float(error_threshold_ratio)

        self.packet_len = int(packet_len)
        if self.packet_len < self.sw_len:
            raise ValueError("packet_len must be >= length of sync_word")

        self.shift_registers =[np.zeros(self.sw_len, dtype=np.float32) for _ in range(self.num_channels)]
        self.collecting =[False] * self.num_channels
        self.packet_bufs = [None] * self.num_channels
        self.samples_collected = [0] * self.num_channels

        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUSH)
        zmq_endpoint = f"tcp://{zmq_ip}:{zmq_port}"
        self.zmq_socket.bind(zmq_endpoint)

    def stop(self):
        self.zmq_socket.close()
        self.zmq_context.term()
        return True

    def _start_collection(self, ch, initial_window):
        self.packet_bufs[ch] = np.empty(self.packet_len, dtype=np.float32)
        self.packet_bufs[ch][:self.sw_len] = initial_window
        self.samples_collected[ch] = self.sw_len
        self.collecting[ch] = True

    def _append_symbol(self, ch, s):
        if not self.collecting[ch]:
            return
        
        collected = self.samples_collected[ch]
        if collected < self.packet_len:
            self.packet_bufs[ch][collected] = float(s)
            self.samples_collected[ch] += 1
            
        if self.samples_collected[ch] >= self.packet_len:
            ch_header = struct.pack('<I', ch + 1)
            payload = self.packet_bufs[ch].tobytes()
            
            self.zmq_socket.send(ch_header + payload)
            
            self.collecting[ch] = False
            self.packet_bufs[ch] = None
            self.samples_collected[ch] = 0

    def work(self, input_items, output_items):
        nread = len(input_items[0])

        for ch in range(self.num_channels):
            in_data = input_items[ch]
            
            for i in range(nread):
                x = float(in_data[i])

                if self.collecting[ch]:
                    self._append_symbol(ch, x)
                else:
                    self.shift_registers[ch][:-1] = self.shift_registers[ch][1:]
                    self.shift_registers[ch][-1] = x
                    
                    sse = float(np.sum((self.shift_registers[ch] - self.sync_word)**2))
                    
                    if sse <= self.error_threshold:
                        self._start_collection(ch, self.shift_registers[ch].copy())
        return nread