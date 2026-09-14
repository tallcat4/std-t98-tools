# devices/

機種ごとの参考設定（backend の TOML）です。**機種の既定だけ**を持ち、
個体固有値（周波数校正 `freq_err_offset`、`serial=` など）は持ちません。

デスクトップ GUI は単一の固定設定ファイル（`~/.config/std-t98/settings.toml`）を
フォームで直接編集する方式で、このディレクトリは参照しません。端末から直接使う
場合の出発点として使えます:
`STD_T98_BACKEND_CONFIG=devices/usrp-b210.toml python3 std_t98_multi_service_launcher.py`
（個体固有値は自分のコピーに書き足してください）

各ファイルは通常の backend 設定 TOML（`[sdr]` / `[channelizer]` / `[demod]`）です。
全キーの意味は [`../config.example.toml`](../config.example.toml) を参照してください。

## 収録機種

backend は USRP（UHD）専用です。

| ファイル | 機種 |
| --- | --- |
| `usrp-b210.toml` | Ettus / LibreSDR B210 |

## 機種を追加する

1. 近い機種のファイルをコピーし、判別しやすい名前を付けます（例: `usrp-x310.toml`）。
2. `[sdr]` に機種で必要な項目（`antenna` / `sample_rate` / `gain_element` / `bandwidth` など）を
   設定します。選択肢は `uhd_find_devices`（GUI の Detect SDRs）や `uhd_usrp_probe` で確認できます。
   `gain_element` は機種ごとのUHD named gain stage 名（B210 は `PGA`）です。
3. **個体固有値は入れない**でください。`freq_err_offset`（周波数校正）は無線機ごとに実測
   する値、`device_args` の `serial=...` は個体選択で、いずれも各自の環境依存です。
   テンプレートは機種の既定だけを持ち、これらは利用者が自分のコピーで足します。

`[meta]` などの独自セクションは backend が拒否するため置けません。
