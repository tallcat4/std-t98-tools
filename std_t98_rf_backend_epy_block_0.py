#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gnuradio import gr
import zmq

class sync_word_correlator(gr.sync_block):
    def __init__(self,
                 sync_word=None,
                 threshold_ratio=0.9,
                 packet_len=192,
                 abs_limit=4.0):
        gr.sync_block.__init__(
            self,
            name="Sync Word Correlator",
            in_sig=[np.float32],
            out_sig=None
        )

        if sync_word is None:
            sync_word =[-3.0, +1.0, -3.0, +3.0, -3.0, -3.0, +3.0, +3.0, -1.0, +3.0]
        self.sync_word = np.asarray(sync_word, dtype=np.float32)
        self.sw_len = len(self.sync_word)

        # 同期ワードのエネルギー
        sw_energy = float(np.dot(self.sync_word, self.sync_word))
        
        self.error_threshold = sw_energy * 2.0 * (1.0 - float(threshold_ratio))

        self.shift_register = np.zeros(self.sw_len, dtype=np.float32)

        self.packet_len = int(packet_len)
        if self.packet_len < self.sw_len:
            raise ValueError("packet_len must be >= length of sync_word")
        self.collecting = False
        self.packet_buf = None
        self.samples_collected = 0
        self.abs_limit = float(abs_limit)

        # ZMQ PUSH ソケットの初期化
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_socket.bind("tcp://127.0.0.1:5555")

    def stop(self):
        # ブロック停止時にZMQをクリーンアップ
        self.zmq_socket.close()
        self.zmq_context.term()
        return True

    def _start_collection(self, initial_window):
        self.packet_buf = np.empty(self.packet_len, dtype=np.float32)
        self.packet_buf[:self.sw_len] = initial_window
        self.samples_collected = self.sw_len
        self.collecting = True

    def _append_symbol(self, s):
        if not self.collecting:
            return
        if self.samples_collected < self.packet_len:
            self.packet_buf[self.samples_collected] = float(s)
            self.samples_collected += 1
        if self.samples_collected >= self.packet_len:
            # Numpy配列の生バイト列をそのまま送信
            self.zmq_socket.send(self.packet_buf.tobytes())
            # リセット
            self.collecting = False
            self.packet_buf = None
            self.samples_collected = 0

    def work(self, input_items, output_items):
        in_data = input_items[0]
        nread = len(in_data)

        for i in range(nread):
            x = float(in_data[i])
            # クリップ処理 (abs_limitによる制限)
            if x > self.abs_limit: x = self.abs_limit
            elif x < -self.abs_limit: x = -self.abs_limit

            if self.collecting:
                self._append_symbol(x)
            else:
                self.shift_register[:-1] = self.shift_register[1:]
                self.shift_register[-1] = x
                
                error = float(np.sum((self.shift_register - self.sync_word) ** 2))
                
                if error <= self.error_threshold:
                    self._start_collection(self.shift_register.copy())
                    
        return nread