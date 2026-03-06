#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sounddevice as sd

def main():
    # デバイス一覧を表示
    print("=== 利用可能なオーディオデバイス一覧 ===")
    print(sd.query_devices())
    print("======================================")
    
    # 使用するデバイス番号を入力
    dev_str = input("使用する出力デバイスの番号を入力してください (そのままEnterでデフォルト): ")
    device_id = int(dev_str.strip()) if dev_str.strip() else None

    SAMPLE_RATE = 48000
    FREQ = 440.0         # 440Hz (ラの音)
    AMPLITUDE = 16000    # 音量
    CHUNK_SIZE = 4800    # 書き込みサイズ
    phase = 0

    print(f"\n再生を開始します (デバイス: {device_id if device_id is not None else 'デフォルト'})")
    print("停止するには Ctrl+C を押してください...")

    try:
        # device=device_id で明示的に出力先を指定
        with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', latency='low', device=device_id) as audio_stream:
            while True:
                t = (np.arange(CHUNK_SIZE) + phase) / SAMPLE_RATE
                samples = (AMPLITUDE * np.sin(2 * np.pi * FREQ * t)).astype(np.int16)
                audio_stream.write(samples.tobytes())
                phase += CHUNK_SIZE

    except KeyboardInterrupt:
        print("\n再生を停止しました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()