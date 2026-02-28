import numpy as np

def _H_mag(f, T, alpha):
    af = np.abs(f)
    f0 = (1.0 - alpha) / (2.0 * T)
    f1 = (1.0 + alpha) / (2.0 * T)
    H = np.zeros_like(af)
    core = af < f0
    trans = (af >= f0) & (af <= f1)
    H[core] = 1.0
    H[trans] = np.cos((T/(4.0*alpha))*(2.0*np.pi*af[trans] - np.pi*(1.0-alpha)/T))
    return H

def _P_mag(f, T):
    return np.sinc(f * T)

def _D_mag(f, T):
    P = _P_mag(f, T)
    eps = 1e-8
    return 1.0 / np.clip(np.abs(P), eps, None)

def _design_taps(fs, sym_rate, alpha, ntaps, which="tx", fft_len=None, window=True):
    T = 1.0 / sym_rate
    if fft_len is None:
        p = int(np.ceil(np.log2(max(2048, 8*ntaps))))
        fft_len = 1 << p

    f = np.fft.fftfreq(fft_len, d=1.0/fs)
    f = np.fft.fftshift(f)

    H = _H_mag(f, T, alpha)

    if which == "tx":
        S = H * _P_mag(f, T)
    elif which == "rx":
        S = H * _D_mag(f, T)
    else:
        raise ValueError("which must be 'tx' or 'rx'")

    h_circ = np.fft.ifft(np.fft.ifftshift(S)).real
    h_centered = np.fft.fftshift(h_circ)
    mid = fft_len // 2
    half = ntaps // 2
    taps = h_centered[mid - half : mid + half + 1] if (ntaps % 2 == 1) \
        else h_centered[mid - half : mid + half]

    if window:
        beta = 8.0
        w = np.kaiser(len(taps), beta)
        taps = taps * w

    g = np.sum(taps)
    if g != 0:
        taps = taps / g

    return taps.astype(np.float32).tolist()

def _save_taps_inline(taps, outfile):
    """[a, b, c, ...] 形式で改行なし保存"""
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("[" + ", ".join(f"{x:.15g}" for x in taps) + "]")
    print(f"[saved inline] {outfile} ({len(taps)} taps)")

def make_tx_taps(fs, sym_rate=2400.0, alpha=0.2, ntaps=257, outfile=None):
    taps = _design_taps(fs, sym_rate, alpha, ntaps, which="tx")
    if outfile:
        _save_taps_inline(taps, outfile)
    return taps

def make_rx_taps(fs, sym_rate=2400.0, alpha=0.2, ntaps=257, outfile=None):
    taps = _design_taps(fs, sym_rate, alpha, ntaps, which="rx")
    if outfile:
        _save_taps_inline(taps, outfile)
    return taps

if __name__ == "__main__":
    fs = 96000
    sym_rate = 2400
    alpha = 0.2
    sps = int(fs / sym_rate)
    ntaps = 20 * sps  
    print(f"fs={fs}, sym_rate={sym_rate}, sps={sps}, ntaps={ntaps}")

    tx_taps = make_tx_taps(fs=fs, sym_rate=sym_rate, alpha=alpha, ntaps=ntaps, outfile="tx_taps.txt")
    rx_taps = make_rx_taps(fs=fs, sym_rate=sym_rate, alpha=alpha, ntaps=ntaps, outfile="rx_taps.txt")


    print(f"tx_taps: {len(tx_taps)} samples, rx_taps: {len(rx_taps)} samples")