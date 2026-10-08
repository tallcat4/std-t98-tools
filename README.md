# std-t98-tools

ARIB STD-T98(デジタル簡易無線、351 MHz 帯)を USRP で受信し、30 チャンネルを同時に復調・復号して音声を再生する GNU Radio / Python ツール群です。

## 主な機能

- **30 チャンネル同時受信** — 全チャンネルを並列に復調し、送信があったチャンネルの音声をそれぞれ再生します。
- **プロトコル解析** — RICH / SACCH / PICH / TCH を解析し、チャンネルごとの受信状況を表示します。
- **AMBE+2 音声の復号** — 受信した音声をその場でデコードして再生します。
- **秘話の自動解除** — 学習済みモデルで秘話コードを推定し、スクランブルされた音声を復元します。
- **デスクトップ GUI** — 30 チャンネルのタイル表示、スケルチなどの調整、USRP の状態表示。端末用のダッシュボードもあります。
- **録音と再生** — IQ を録音して後から解析でき、SDR なしで全体の動作を確認できます。

## 必要な機材

| 機材 | 内容 |
| --- | --- |
| SDR | Ettus Research USRP(UHD 対応機。USRP B210 で動作を確認) |
| アンテナ | 351 MHz 帯に合ったもの |
| PC | Linux が動く PC |

## 動作環境

- Linux(Arch Linux で動作を確認)
- Python 3.11 以降
- GNU Radio(`gnuradio.uhd` を含む)、UHD

## インストール

GNU Radio と UHD はディストリビューションのパッケージで入れます。

```bash
# Debian / Ubuntu の例
sudo apt install gnuradio gnuradio-dev libuhd-dev uhd-host libportaudio2
sudo uhd_images_downloader
```

残りの依存は `setup.sh` が `env/` に入れます。root 権限は不要です。

```bash
./setup.sh                 # 全機能
./setup.sh --no-secret     # 秘話解除(torch)を除く
```

## 使い方

USRP を接続して起動します。

```bash
python3 -m app                               # デスクトップ GUI
python3 std_t98_multi_service_launcher.py    # 端末で起動
```

送信を受けると、そのチャンネルの音声が再生されます。`./install-desktop.sh` を実行すると、アプリメニューから起動できるようになります。

SDR の設定(ゲイン、アンテナ端子、周波数校正など)、診断ツール、トラブルシューティングは [詳細マニュアル](docs/manual.md) を参照してください。

## 関連リポジトリ

- [std-t98-vocode](https://github.com/tallcat4/std-t98-vocode) — AMBE+2 音声ファイルのデコード・秘話の復号
- [pyambelib](https://github.com/tallcat4/pyambelib) — AMBE 復号ライブラリ(mbelib-neo のラッパー)

## ライセンス

GNU General Public License v3.0。詳細は [`LICENSE`](LICENSE) を参照してください。
