# recv_float_rich_decode.py
import zmq
import numpy as np
import pmt
import time
import collections
import os

def decode_rich(rich_bits_str: str) -> dict:
    """
    RICH (Radio Information Channel) の復号と情報抽出を行う。
    
    [入力]
      rich_bits_str: 16ビットの2進数文字列 (例: "010011...")
      
    [処理詳細]
      1. 軟判定マッピング (16 bits -> 8 bits)
         受信した16ビットを2ビットずつのペアに分割し、以下のテーブルに基づき8ビットの情報にデコードする。
         "00", "01" -> 0
         "10", "11" -> 1
         
      2. フィールド分解
         復元された8ビットを以下の構成に従って分解する。
         - F      (1 bit) : バースト種別 (0=同期バースト, 1=通信用バースト)
         - Res1   (1 bit) : 予約ビット1
         - Res2   (1 bit) : 予約ビット2
         - M      (3 bits): モード情報 (ビット結合して数値化)
         - D      (1 bit) : データフラグ
         - Parity (1 bit) : 偶数パリティビット
         
      3. パリティチェック
         先頭7ビットの総和の偶奇(XOR)を計算し、8ビット目のParityと照合する。
         
    [戻り値]
      抽出された各フィールド(F, Res1, Res2, M, D, Parity)と、
      パリティチェック結果(Parity_OK: bool)を含む辞書。
    """
    if len(rich_bits_str) != 16:
        raise ValueError(f"RICH must be exactly 16 bits, got {len(rich_bits_str)} bits.")
        
    rich_decoded_bits =[]
    for i in range(0, len(rich_bits_str), 2):
        pair = rich_bits_str[i:i+2]
        if pair in ("00", "01"):
            rich_decoded_bits.append(0)
        elif pair in ("10", "11"):
            rich_decoded_bits.append(1)
        else:
            raise ValueError(f"Invalid 2-bit pair '{pair}' in RICH bits.")

    val_f    = rich_decoded_bits[0]
    val_res1 = rich_decoded_bits[1]
    val_res2 = rich_decoded_bits[2]
    val_m    = (rich_decoded_bits[3] << 2) | (rich_decoded_bits[4] << 1) | rich_decoded_bits[5]
    val_d    = rich_decoded_bits[6]
    val_p    = rich_decoded_bits[7]

    calc_parity = sum(rich_decoded_bits[:7]) % 2
    parity_ok = (calc_parity == val_p)

    return {
        "F": val_f, "Res1": val_res1, "Res2": val_res2,
        "M": val_m, "D": val_d, "Parity": val_p, "Parity_OK": parity_ok
    }

def decode_sacch(sacch_bits_str: str) -> dict:
    """
    SACCH (Slow Associated Control Channel) の復号と情報抽出を行う。
    
    [入力]
      sacch_bits_str: 60ビットの2進数文字列
      
    [処理詳細]
      1. デインターリーブ (60 bits -> 60 bits)
         送信時の 12列 x 5行 のブロックインターリーバの逆操作を行う。
         （縦方向に書き込まれたデータを、横方向に読み出して元の順序に戻す）
         
      2. デパンクチャ (60 bits -> 72 bits)
         パンクチャリング行列[G1: 1 1 1 1 1 1, G2: 1 1 0 1 1 0] に従い、
         送信時に間引かれた G2 のインデックス 2 と 5 のタイミングに
         消失ビット(None)を挿入し、本来の畳み込み符号長(72ビット)に復元する。
         
      3. ビタビ復号 (72 bits -> 36 bits)
         拘束長 K=5 (状態数16)、符号化率 R=1/2 の畳み込み符号を復号する。
         生成多項式: G1(D) = 1+D^3+D^4, G2(D) = 1+D+D^2+D^4
         ※最後の4ビット(Tail Bits)は "0" 固定である仕様を利用し、
           終端のパス遷移を 0 のみに制限することで強力にエラーを訂正する。
           
      4. CRCチェック (26 bits + 6 bits)
         抽出した36ビット(情報26 + CRC6 + Tail4)のうち、先頭26ビットから
         生成多項式: 1 + X + X^2 + X^5 + X^6 にて6ビットCRCを算出し照合する。
         
      5. 情報抽出 (26 bits -> 辞書)
         F(1bit), Wr(2bits), MsgType(5bits), CallStat(2bits), 
         UserCode(9bits), MakerCode(7bits) に分解する。
         
    [戻り値]
      各種メッセージフィールド、CRC結果(CRC_OK)、推定訂正エラー数(BitErrors)を含む辞書。
    """
    if len(sacch_bits_str) != 60:
        raise ValueError(f"SACCH must be exactly 60 bits, got {len(sacch_bits_str)} bits.")
        
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
            x_t  = bit
            x_t1 = (state >> 3) & 1
            x_t2 = (state >> 2) & 1
            x_t3 = (state >> 1) & 1
            x_t4 = state & 1
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
    corrected_errors = metrics[0][0]

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
            crc_ok = True; crc_calculated = crc_calculated_rev

    def bits_to_int(bit_list: list) -> int:
        return int("".join(map(str, bit_list)), 2)

    return {
        "F": msg_bits[0],
        "Wr": bits_to_int(msg_bits[1:3]),
        "MsgType": bits_to_int(msg_bits[3:8]),
        "CallStat": bits_to_int(msg_bits[8:10]),
        "UserCode": bits_to_int(msg_bits[10:19]),
        "MakerCode": bits_to_int(msg_bits[19:26]),
        "CRC_OK": crc_ok,
        "CRC_Recv": "".join(map(str, crc_bits_received)),
        "CRC_Calc": "".join(map(str, crc_calculated)),
        "BitErrors": corrected_errors
    }

def decode_pich(pich_bits_str: str) -> dict:
    """
    PICH (Parameter Information Channel) の復号と情報抽出を行う。
    同期バースト(F=0)時のTCH1領域(144bits)がこれに該当する。[入力]
      pich_bits_str: 144ビットの2進数文字列
      
    [処理詳細]
      1. デインターリーブ (144 bits -> 144 bits)
         16列 x 9行 のブロックインターリーバの逆操作を行う。
         
      2. デパンクチャ (144 bits -> 192 bits)
         パンクチャリング行列 [G1: 1 1, G2: 0 1] に従い、
         出力4ビットあたり1ビット（G2の先頭出力）が間引かれているため、
         該当位置に消失ビット(None)を挿入し、本来の192ビット長に復元する。
         
      3. ビタビ復号 (192 bits -> 96 bits)
         SACCHと同一の 拘束長 K=5, 符号化率 R=1/2 の畳み込み符号を復号する。
         Tail Bits (終端4ビット) が "0" 固定である制約を利用し訂正能力を高める。
         
      4. CRCチェック (80 bits + 12 bits)
         抽出した96ビット(情報80 + CRC12 + Tail4)のうち、先頭80ビットから
         生成多項式: 1 + X + X^2 + X^3 + X^11 + X^12 にて12ビットCRCを算出し照合する。
         
      5. 情報抽出 (80 bits -> CSM + Reserve)
         - CSM (36 bits): 4ビットずつ Binary Coded Decimal (BCD) として解釈し、
                          9桁の10進数文字列に変換する。（ノイズ対応として16進文字も許容）
         - Reserve (44 bits): そのままのビット文字列として抽出。
         
    [戻り値]
      CSM(9桁の文字列)、Reserve、CRC結果(CRC_OK)、推定訂正エラー数(BitErrors)を含む辞書。
    """
    if len(pich_bits_str) != 144:
        raise ValueError(f"PICH must be exactly 144 bits, got {len(pich_bits_str)}")
        
    pich_bits =[int(b) for b in pich_bits_str]
    
    deinterleaved = [0] * 144
    for col in range(16):
        for row in range(9):
            deinterleaved[row * 16 + col] = pich_bits[col * 9 + row]
            
    depunctured =[]
    idx = 0
    for _ in range(48):
        depunctured.append(deinterleaved[idx])
        idx += 1
        depunctured.append(None)
        
        depunctured.append(deinterleaved[idx])
        idx += 1
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
    corrected_errors = metrics[0][0]

    msg_bits = decoded_bits[:80]
    crc_bits_received = decoded_bits[80:92]
    
    s = [1] * 12
    for b in msg_bits:
        fb = b ^ s[11]
        s_next = [0] * 12
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
    if not crc_ok:
        if s == crc_bits_received:
            crc_ok = True; crc_calculated = s

    csm_bits = msg_bits[:36]
    csm_str = ""
    for i in range(9):
        digit_bits = csm_bits[i*4 : i*4+4]
        digit_val = int("".join(map(str, digit_bits)), 2)
        csm_str += f"{digit_val:X}"
        
    reserve_bits = msg_bits[36:80]
    
    return {
        "CSM": csm_str,
        "Reserve": "".join(map(str, reserve_bits)),
        "CRC_OK": crc_ok,
        "CRC_Recv": "".join(map(str, crc_bits_received)),
        "CRC_Calc": "".join(map(str, crc_calculated)),
        "BitErrors": corrected_errors
    }

def format_binary_string(bin_str, chunk_size=8):
    return " ".join([bin_str[i:i+chunk_size] for i in range(0, len(bin_str), chunk_size)])

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

def main():
    ctx = None
    sock = None
    sync_word =[-3, +1, -3, +3, -3, -3, +3, +3, -1, +3]

    AMBE_FILE = "output.ambe"

    open(AMBE_FILE, "wb").close()

    frame_structure =[
        ("SW",    20),
        ("RICH",  16),
        ("SACCH", 60),
        ("TCH1",  144),
        ("TCH2",  144)
    ]

    try:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PULL)
        sock.connect("tcp://127.0.0.1:5555")

        last_recv_time = time.time()
        dt_history = collections.deque(maxlen=2)

        print("Waiting for Float PDU... (Press Ctrl+C to exit)")
        
        while True:
            frames = sock.recv_multipart()
            
            current_time = time.time()
            dt = (current_time - last_recv_time) * 1000
            last_recv_time = current_time

            dt_history.append(dt)
            avg_dt = sum(dt_history) / len(dt_history)
            
            serialized_pdu = frames[-1]
            pdu = pmt.deserialize_str(serialized_pdu)
            payload_pmt = pmt.cdr(pdu)
            
            raw_floats = np.array(pmt.to_python(payload_pmt), dtype=np.float32)

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
            for name, bit_len in frame_structure:
                section_bits = full_binary_str[current_bit_idx : current_bit_idx + bit_len]
                fields[name] = section_bits
                current_bit_idx += bit_len

            print(f"\n====================")
            print(f"Average dt: {avg_dt:.2f} ms | Total Symbols: {symbols.size}")
            
            # --- SW ---
            print(f"SW    ({len(fields['SW']):>3} bits): {format_binary_string(fields['SW'])}")

            # --- RICH ---
            print(f"RICH  ({len(fields['RICH']):>3} bits): {format_binary_string(fields['RICH'])}")
            rich_data = None
            try:
                rich_data = decode_rich(fields['RICH'])
                parity_status = "OK" if rich_data["Parity_OK"] else "NG"
                burst_type = "Sync Burst" if rich_data['F'] == 0 else "Traffic Burst"
                print(f"    --> F      : {rich_data['F']} ({burst_type})")
                print(f"    --> Res    : {rich_data['Res1']}, {rich_data['Res2']}")
                print(f"    --> M      : {rich_data['M']:03b}")
                print(f"    --> D      : {rich_data['D']}")
                print(f"    --> Parity : {rich_data['Parity']} ({parity_status})")
            except Exception as e:
                print(f"    --> RICH Decode Error: {e}")

            # --- SACCH ---
            print(f"SACCH ({len(fields['SACCH']):>3} bits): {format_binary_string(fields['SACCH'])}")
            try:
                sacch_data = decode_sacch(fields['SACCH'])
                crc_status = "OK" if sacch_data["CRC_OK"] else "NG"
                print(f"    --> F          : {sacch_data['F']}")
                print(f"    --> Wr         : {sacch_data['Wr']:02b}")
                print(f"    --> Msg Type   : {sacch_data['MsgType']:05b}")
                
                stat_desc = "Normal" if sacch_data['CallStat'] == 0 else "Secret" if sacch_data['CallStat'] == 1 else "Unknown"
                print(f"    --> Call Stat  : {sacch_data['CallStat']:02b} ({stat_desc})")
                
                print(f"    --> User Code  : {sacch_data['UserCode']}")
                print(f"    --> Maker Code : {sacch_data['MakerCode']}")
                print(f"    --> CRC        : {crc_status} (Recv: {sacch_data['CRC_Recv']}, Calc: {sacch_data['CRC_Calc']})")
                print(f"    --> Bit Errors : {sacch_data['BitErrors']} bits corrected")
            except Exception as e:
                print(f"    --> SACCH Decode Error: {e}")

            # --- TCH1 / TCH2 (Fフラグに基づく分岐) ---
            print(f"TCH1  ({len(fields['TCH1']):>3} bits): {format_binary_string(fields['TCH1'])}")
            print(f"TCH2  ({len(fields['TCH2']):>3} bits): {format_binary_string(fields['TCH2'])}")

            if rich_data is not None:
                # ====== 同期バースト (F=0) の処理 ======
                if rich_data['F'] == 0:
                    print("    -->[Mode: Sync Burst] Processing TCH1 as PICH...")
                    try:
                        pich_data = decode_pich(fields['TCH1'])
                        pich_crc_status = "OK" if pich_data["CRC_OK"] else "NG"
                        
                        print(f"    --> [PICH] CSM        : {pich_data['CSM']}")
                        print(f"    --> [PICH] Reserve    : {pich_data['Reserve']}")
                        print(f"    --> [PICH] CRC        : {pich_crc_status} (Recv: {pich_data['CRC_Recv']}, Calc: {pich_data['CRC_Calc']})")
                        print(f"    -->[PICH] Bit Errors : {pich_data['BitErrors']} bits corrected")
                    except Exception as e:
                        print(f"    --> PICH Decode Error: {e}")
                    
                    print("    --> [Mode: Sync Burst] TCH2 is discarded (Not Used).")

                # ====== 通信用バースト (F=1) の処理 ======
                elif rich_data['F'] == 1:
                    print("    --> [Mode: Traffic Burst] Processing TCH as Voice Data...")
                    
                    # 144bit を 前半72bit / 後半72bit に分割
                    tch1_1 = fields['TCH1'][:72]
                    tch1_2 = fields['TCH1'][72:]
                    tch2_1 = fields['TCH2'][:72]
                    tch2_2 = fields['TCH2'][72:]
                    
                    # 順番にリストに格納
                    blocks =[tch1_1, tch1_2, tch2_1, tch2_2]
                    
                    # バイナリファイルに追記出力
                    with open(AMBE_FILE, "ab") as f_ambe:
                        for block in blocks:
                            # 72文字のビット文字列 ("0101...") を 9バイトのバイナリに変換 (Big-Endian)
                            block_bytes = int(block, 2).to_bytes(9, byteorder='big')
                            
                            # 先頭に 0x48 (10進数で72) を付与して合計10バイトを書き込む
                            f_ambe.write(b'\x48' + block_bytes)
                            
                    print(f"    --> Appended 4 AMBE frames (10 bytes each) to '{AMBE_FILE}'")
            
            print("=" * 30)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        print("Cleaning up...")
        if sock: sock.close()
        if ctx: ctx.term()

if __name__ == "__main__":
    main()