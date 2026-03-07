#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zmq
import numpy as np
import sounddevice as sd
from pyambelib import AmbeDecoder, fec_demod
import sys

# ==========================================
# 復号用キー (LFSR初期状態: 0 - 32767)
# ==========================================
DECRYPTION_KEY = 0

# ==========================================
# 各種テーブル・定数
# ==========================================
_THUMBDV_MAP =[
     0, 18, 36,  1, 19, 37,  2, 20, 38,
     3, 21, 39,  4, 22, 40,  5, 23, 41,
     6, 24, 42,  7, 25, 43,  8, 26, 44,
     9, 27, 45, 10, 28, 46, 11, 29, 47,
    12, 30, 48, 13, 31, 14, 32, 15, 33,
    16, 34, 17, 35
]

FRAME_STRUCTURE =[
    ("SW",    20),
    ("RICH",  16),
    ("SACCH", 60),
    ("TCH1",  144),
    ("TCH2",  144)
]

# ==========================================
# データ処理関数
# ==========================================
def dewhiten(symbol_seq):
    if len(symbol_seq) < 192:
        return symbol_seq
    lfsr =[0, 1, 1, 1, 0, 0, 1, 0, 0]
    whitening_seq =[]
    for _ in range(182):
        s0 = lfsr[-1]
        s4 = lfsr[4]
        new_s8 = s4 ^ s0
        whitening_seq.append(s0)
        lfsr = [new_s8] + lfsr[:-1]
    signs = np.array([1 if b == 0 else -1 for b in whitening_seq], dtype=np.int8)
    output_seq = symbol_seq.copy()
    output_seq[10:] = output_seq[10:] * signs
    return output_seq

def decode_rich(rich_bits_str: str) -> dict:
    rich_decoded_bits =[]
    for i in range(0, len(rich_bits_str), 2):
        pair = rich_bits_str[i:i+2]
        if pair in ("00", "01"):
            rich_decoded_bits.append(0)
        elif pair in ("10", "11"):
            rich_decoded_bits.append(1)
        else:
            raise ValueError(f"Invalid 2-bit pair '{pair}'")
    val_f = rich_decoded_bits[0]
    return {"F": val_f}

def decode_sacch(sacch_bits_str: str) -> dict:
    if len(sacch_bits_str) != 60:
        return {"CRC_OK": False}
    sacch_bits =[int(b) for b in sacch_bits_str]
    
    deinterleaved =[0] * 60
    for col in range(12):
        for row in range(5):
            deinterleaved[row * 12 + col] = sacch_bits[col * 5 + row]
            
    depunctured =[]
    idx = 0
    for _ in range(6):
        for p in range(6):
            depunctured.append(deinterleaved[idx])
            idx += 1
            if p in (2, 5):
                depunctured.append(None)
            else:
                depunctured.append(deinterleaved[idx])
                idx += 1

    trellis = {}
    for state in range(16):
        for bit in (0, 1):
            x_t = bit; x_t1 = (state >> 3) & 1; x_t2 = (state >> 2) & 1
            x_t3 = (state >> 1) & 1; x_t4 = state & 1
            next_state = (bit << 3) | (state >> 1)
            g1_out = x_t ^ x_t3 ^ x_t4
            g2_out = x_t ^ x_t1 ^ x_t2 ^ x_t4
            trellis[(state, bit)] = (next_state, g1_out, g2_out)

    metrics = {s: (float('inf'),[]) for s in range(16)}
    metrics[0] = (0,[])
    
    for t in range(36):
        r1 = depunctured[t * 2]
        r2 = depunctured[t * 2 + 1]
        new_metrics = {s: (float('inf'),[]) for s in range(16)}
        valid_bits = (0, 1) if t < 32 else (0,)
        for state, (cost, hist) in metrics.items():
            if cost == float('inf'): continue
            for bit in valid_bits:
                next_state, g1_out, g2_out = trellis[(state, bit)]
                dist = 0
                if r1 is not None and r1 != g1_out: dist += 1
                if r2 is not None and r2 != g2_out: dist += 1
                new_cost = cost + dist
                if new_cost < new_metrics[next_state][0]:
                    new_metrics[next_state] = (new_cost, hist + [bit])
        metrics = new_metrics
        
    decoded_bits = metrics[0][1]
    msg_bits = decoded_bits[:26]
    crc_bits_received = decoded_bits[26:32]
    
    s =[1, 1, 1, 1, 1, 1]
    for b in msg_bits:
        fb = b ^ s[5]
        s =[fb, s[0] ^ fb, s[1] ^ fb, s[2], s[3], s[4] ^ fb]
        
    crc_calculated =[s[5], s[4], s[3], s[2], s[1], s[0]]
    crc_ok = (crc_calculated == crc_bits_received)
    if not crc_ok:
        crc_calculated_rev = [s[0], s[1], s[2], s[3], s[4], s[5]]
        if crc_calculated_rev == crc_bits_received:
            crc_ok = True

    def bits_to_int(bit_list: list) -> int:
        return int("".join(map(str, bit_list)), 2)

    return {
        "UserCode": bits_to_int(msg_bits[10:19]),
        "MakerCode": bits_to_int(msg_bits[19:26]),
        "CallStat": bits_to_int(msg_bits[8:10]),
        "CRC_OK": crc_ok
    }

def decode_pich(pich_bits_str: str) -> dict:
    if len(pich_bits_str) != 144:
        return {"CRC_OK": False}
    pich_bits =[int(b) for b in pich_bits_str]
    
    deinterleaved = [0] * 144
    for col in range(16):
        for row in range(9):
            deinterleaved[row * 16 + col] = pich_bits[col * 9 + row]
            
    depunctured =[]
    idx = 0
    for _ in range(48):
        depunctured.append(deinterleaved[idx]); idx += 1
        depunctured.append(None)
        depunctured.append(deinterleaved[idx]); idx += 1
        depunctured.append(deinterleaved[idx]); idx += 1

    trellis = {}
    for state in range(16):
        for bit in (0, 1):
            x_t = bit; x_t1 = (state >> 3) & 1; x_t2 = (state >> 2) & 1
            x_t3 = (state >> 1) & 1; x_t4 = state & 1
            next_state = (bit << 3) | (state >> 1)
            g1_out = x_t ^ x_t3 ^ x_t4
            g2_out = x_t ^ x_t1 ^ x_t2 ^ x_t4
            trellis[(state, bit)] = (next_state, g1_out, g2_out)

    metrics = {s: (float('inf'),[]) for s in range(16)}
    metrics[0] = (0,[])
    
    for t in range(96):
        r1 = depunctured[t * 2]
        r2 = depunctured[t * 2 + 1]
        new_metrics = {s: (float('inf'),[]) for s in range(16)}
        valid_bits = (0, 1) if t < 92 else (0,)
        for state, (cost, hist) in metrics.items():
            if cost == float('inf'): continue
            for bit in valid_bits:
                next_state, g1_out, g2_out = trellis[(state, bit)]
                dist = 0
                if r1 is not None and r1 != g1_out: dist += 1
                if r2 is not None and r2 != g2_out: dist += 1
                new_cost = cost + dist
                if new_cost < new_metrics[next_state][0]:
                    new_metrics[next_state] = (new_cost, hist + [bit])
        metrics = new_metrics
        
    decoded_bits = metrics[0][1]
    msg_bits = decoded_bits[:80]
    crc_bits_received = decoded_bits[80:92]
    
    s =[1] * 12
    for b in msg_bits:
        fb = b ^ s[11]
        s_next =[0] * 12
        s_next[0] = fb
        s_next[1] = s[0] ^ fb
        s_next[2] = s[1] ^ fb
        s_next[3] = s[2] ^ fb
        s_next[4] = s[3]
        s_next[5] = s[4]
        s_next[6] = s[5]
        s_next[7] = s[6]
        s_next[8] = s[7]
        s_next[9] = s[8]
        s_next[10] = s[9]
        s_next[11] = s[10] ^ fb
        s = s_next
        
    crc_calculated = s[::-1]
    crc_ok = (crc_calculated == crc_bits_received)
    if not crc_ok and s == crc_bits_received:
        crc_ok = True

    csm_bits = msg_bits[:36]
    csm_str = ""
    for i in range(9):
        digit_bits = csm_bits[i*4 : i*4+4]
        digit_val = int("".join(map(str, digit_bits)), 2)
        csm_str += f"{digit_val:X}"
        
    return {
        "CSM": csm_str,
        "CRC_OK": crc_ok
    }

def bits_to_bytes(bits):
    result =[]
    for i in range(0, len(bits), 8):
        byte_val = 0
        chunk_len = min(8, len(bits) - i)
        for j in range(chunk_len):
            byte_val = (byte_val << 1) | bits[i + j]
        if chunk_len < 8:
            byte_val <<= (8 - chunk_len)
        result.append(byte_val)
    return result

def fec_demod_to_2450_payload(payload_3600: bytes) -> bytes:
    try:
        bits = fec_demod(payload_3600)
    except Exception:
        return b'\x00' * 7
    bytes_from_bits = bits_to_bytes(bits)
    if len(bytes_from_bits) >= 7:
        return bytes(bytes_from_bits[:7])
    else:
        return b'\x00' * 7

def generate_pn_sequence_196(initial_state):
    state = initial_state
    pn_bits =[]
    for _ in range(196):
        output = state & 1
        pn_bits.append(output)
        s0 = state & 1
        s1 = (state >> 1) & 1
        feedback = s0 ^ s1
        state = state >> 1
        if feedback:
            state |= (1 << 14)
    return pn_bits

def descramble_burst(payloads, key_196):
    result =[]
    for frame_idx in range(4):
        payload = payloads[frame_idx]
        key_slice = key_196[frame_idx * 49 : (frame_idx + 1) * 49]

        raw_bits =[]
        for byte_val in payload:
            for bit_pos in range(7, -1, -1):
                raw_bits.append((byte_val >> bit_pos) & 1)
        raw_bits = raw_bits[:49]

        descrambled_bits =[raw_bits[j] ^ key_slice[_THUMBDV_MAP[j]] for j in range(49)]

        val = 0
        for bit in descrambled_bits:
            val = (val << 1) | bit
        val <<= 7
        result.append(val.to_bytes(7, byteorder='big'))
    return result

# ==========================================
# コンソール表示 (インプレース更新)
# ==========================================
def print_dashboard(rx_status, csm, sacch, audio_status):
    # カーソルを上に7行移動して上書き (RXインジケーター分を1行追加)
    sys.stdout.write("\033[7A")
    sys.stdout.write("\033[K" + "-"*48 + "\n")
    sys.stdout.write("\033[K[STD-T98 Real-time Receiver Dashboard]\n")
    
    # RXインジケーターの装飾
    if rx_status == "OPEN":
        rx_color = "\033[1;32m OPEN  \033[0m" # 緑色太字
    else:
        rx_color = "\033[1;30m CLOSE \033[0m" # 暗いグレー

    sys.stdout.write(f"\033[K  > RX Indicator : [{rx_color}]\n")
    sys.stdout.write(f"\033[K  > CSM (PICH)   : {csm if csm else 'Waiting...'}\n")
    
    sacch_str = "Waiting..."
    if sacch:
        stat_val = sacch.get('CallStat', 0)
        stat_desc = "Normal" if stat_val == 0 else "Secret" if stat_val == 1 else str(stat_val)
        sacch_str = f"User: {sacch.get('UserCode',0):03d} | Maker: {sacch.get('MakerCode',0):03d} | Stat: {stat_desc}"
        
    sys.stdout.write(f"\033[K  > SACCH        : {sacch_str}\n")
    sys.stdout.write(f"\033[K  > Audio Status : {audio_status}\n")
    sys.stdout.write("\033[K" + "-"*48 + "\n")
    sys.stdout.flush()

# ==========================================
# メイン処理
# ==========================================
def main():
    print("Initializing components...")
    
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.connect("tcp://127.0.0.1:5555")

    # ポーラーの初期化 (100msでタイムアウトさせるために使用)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    decoder = AmbeDecoder()
    key_196 = generate_pn_sequence_196(DECRYPTION_KEY)

    print(f"Decryption Key (LFSR Init) set to: {DECRYPTION_KEY}")
    print("Waiting for ZMQ stream... (Press Ctrl+C to stop)\n")
    
    # ダッシュボード用のスペース(7行分)を空けておく
    print("\n" * 7, end="")

    SAMPLE_RATE = 48000
    UPSAMPLE_FACTOR = 6
    
    # 状態保持用変数
    last_rx_status = None
    last_csm = None
    last_sacch = None

    try:
        with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', latency='low', device="pipewire") as audio_stream:
            
            # 初期描画
            print_dashboard("CLOSE", last_csm, last_sacch, "Listening...")
            last_rx_status = "CLOSE"
            
            while True:
                # 100msのタイムアウト付きでソケットの受信を待機
                events = dict(poller.poll(100))
                
                # イベントがある場合 (データ受信中)
                if sock in events:
                    raw_bytes = sock.recv()
                    rx_status = "OPEN"
                    
                    raw_floats = np.frombuffer(raw_bytes, dtype=np.float32)

                    if len(raw_floats) != 192:
                        continue

                    symbols = np.zeros(raw_floats.shape, dtype=np.int8)
                    symbols[raw_floats > 2.0] = 3
                    symbols[(raw_floats > 0.0) & (raw_floats <= 2.0)] = 1
                    symbols[(raw_floats > -2.0) & (raw_floats <= 0.0)] = -1
                    symbols[raw_floats <= -2.0] = -3

                    symbols = dewhiten(symbols)
                    
                    mapped_bits_val = np.zeros_like(symbols, dtype=np.uint8)
                    mapped_bits_val[symbols == -3] = 3
                    mapped_bits_val[symbols == -1] = 2
                    mapped_bits_val[symbols ==  1] = 0
                    mapped_bits_val[symbols ==  3] = 1
                    full_binary_str = "".join([f"{val:02b}" for val in mapped_bits_val])

                    fields = {}
                    current_bit_idx = 0
                    for name, bit_len in FRAME_STRUCTURE:
                        fields[name] = full_binary_str[current_bit_idx : current_bit_idx + bit_len]
                        current_bit_idx += bit_len

                    try:
                        rich_data = decode_rich(fields['RICH'])
                    except Exception:
                        continue

                    # ===================================================
                    # 1. 同期バースト (F=0) の処理
                    # ===================================================
                    if rich_data['F'] == 0:
                        pich_data = decode_pich(fields['TCH1'])
                        
                        if pich_data.get("CRC_OK"):
                            last_csm = pich_data.get("CSM")
                            
                        print_dashboard(rx_status, last_csm, last_sacch, "Sync Burst (No Audio)")

                    # ===================================================
                    # 2. 通信用バースト (F=1) の処理
                    # ===================================================
                    elif rich_data['F'] == 1:
                        sacch_data = decode_sacch(fields['SACCH'])
                        
                        if sacch_data.get("CRC_OK"):
                            last_sacch = sacch_data
                            
                        print_dashboard(rx_status, last_csm, last_sacch, "Traffic Burst (Playing Audio...)")
                        
                        # 音声デコード＆出力処理
                        tch1_1, tch1_2 = fields['TCH1'][:72], fields['TCH1'][72:]
                        tch2_1, tch2_2 = fields['TCH2'][:72], fields['TCH2'][72:]
                        blocks_bin =[tch1_1, tch1_2, tch2_1, tch2_2]

                        payloads_3600 =[int(b, 2).to_bytes(9, byteorder='big') for b in blocks_bin]
                        payloads_2450 =[fec_demod_to_2450_payload(p) for p in payloads_3600]
                        descrambled_payloads = descramble_burst(payloads_2450, key_196)

                        pcm_out = b''
                        for payload in descrambled_payloads:
                            samples = decoder.decode_2450(payload)
                            if samples:
                                audio_array_8k = np.array(samples, dtype=np.int16)
                                audio_array_48k = np.repeat(audio_array_8k, UPSAMPLE_FACTOR)
                                pcm_out += audio_array_48k.tobytes()
                        
                        if pcm_out:
                            audio_stream.write(pcm_out)

                    last_rx_status = rx_status

                # イベントがない場合 (100msタイムアウト)
                else:
                    rx_status = "CLOSE"
                    
                    # 状態がOPENからCLOSEに変わった瞬間だけ画面を更新
                    if rx_status != last_rx_status:
                        print_dashboard(rx_status, last_csm, last_sacch, "Idle")
                        last_rx_status = rx_status
                    
                    continue

    except KeyboardInterrupt:
        sys.stdout.write("\n" * 7)
        print("Playback stopped by user.")
    except Exception as e:
        sys.stdout.write("\n" * 7)
        print(f"Stream error: {e}")
    finally:
        sock.close()
        ctx.term()

if __name__ == "__main__":
    main()