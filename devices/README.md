# devices/

機種ごとの設定プリセット（backend の TOML）です。GUI の Device プルダウンに並び、
選ぶと解決結果がプレビューされ、そのまま起動に使えます。端末なら
`STD_T98_BACKEND_CONFIG=devices/usrp-b210.toml python3 std_t98_multi_service_launcher.py`
のように渡せます。

各ファイルは通常の backend 設定 TOML（`[sdr]` / `[channelizer]` / `[demod]`）です。
全キーの意味は [`../config.example.toml`](../config.example.toml) を参照してください。

## 収録機種

| ファイル | 機種 | driver |
| --- | --- | --- |
| `usrp-b210.toml` | Ettus / LibreSDR B210 | `uhd` |
| `rtl-sdr.toml` | RTL2832U ドングル | `rtlsdr` |

## 機種を追加する

1. 近い機種のファイルをコピーし、判別しやすい名前を付けます（例: `hackrf.toml`）。
2. **先頭のコメント行が Device プルダウンの表示名**になります。2 行目以降のコメントは説明として扱われます。
3. `[sdr]` の `driver` を SoapySDR のドライバキーにし、機種で必要な項目
   （`antenna` / `sample_rate` / `gain_element` / `bandwidth` など）を設定します。
   選択肢は `SoapySDRUtil --find`（GUI の Detect SDRs）や `--probe` で確認できます。
4. **個体固有値は入れない**でください。`freq_err_offset`（周波数校正）は無線機ごとに実測
   する値、`device_args` の `serial=...` は個体選択で、いずれも各自の環境依存です。
   プリセットは機種の既定だけを持ち、これらは利用者が自分のコピーで足します。

`[meta]` などの独自セクションは backend が拒否するため置けません。表示名・説明は
先頭コメントで表現します。
