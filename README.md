# std-t98-tools

ARIB STD-T98 のマルチチャネル受信系だけを残した GNU Radio / Python ツール群です。

## 現在のスコープ

このリポジトリは、`std_t98_multi_service_launcher.py` を起点にした 30ch 受信スタックだけを保持します。単一チャネル用エントリポイント、ファイル出力専用デコーダ、旧互換ラッパは削除済みです。

実行時の構成は次の 3 プロセスです。

- `std_t98_30ch_multi_rf_backend.py`
  - GNU Radio 側で 30ch の復調と同期語検出を行い、`FramePacket` を UDS へ送ります。
- `std_t98_multi_protocol_service.py`
  - `FramePacket` を復号し、RICH / SACCH / PICH を dashboard 向けに整形しつつ、TCH を `VoiceBurstPacket` として audio service に渡します。
- `std_t98_multi_audio_service.py`
  - `VoiceBurstPacket` を受けて pyambelib で復号し、PCM を再生します。

`std_t98_multi_service_launcher.py` はこれらをまとめて起動し、中央 dashboard と status 集約を担当します。

## 主要モジュール

- `std_t98_multi_service_launcher.py`
  - backend / protocol / audio の起動順制御、Python 実行環境の自動選択、dashboard 更新。
- `std_t98_multi_sync.py`
  - GNU Radio Embedded Python block。各チャンネルの同期語相関を行い、`FramePacket` を送信。
- `core/protocol/`
  - デホワイトニング、フレーム分解、RICH / SACCH / PICH / TCH 処理。
- `core/audio/` と `core/crypto/`
  - AMBE 変換、秘話復号、音声再生用処理。
- `ipc/message_schema.py`
  - `FramePacket`、`VoiceBurstPacket`、`StatusPacket` のバイナリ schema。
- `ipc/transport/uds_seqpacket.py`
  - Unix Domain Socket の `SOCK_SEQPACKET` transport。
- `firdes.py`
  - バックエンドで使うフィルタタップ生成。

## 動作要件

- Linux
- GNU Radio / SoapySDR / RTL-SDR が使える Python 環境
- `pyambelib` と `sounddevice` が使える Python 環境

launcher は backend 用 Python と service 用 Python を別々に解決できます。既定では service 側に `env/bin/python`、backend 側にシステム Python を優先します。

## 使い方

フルスタック起動:

```bash
./env/bin/python std_t98_multi_service_launcher.py
```

既に backend を別プロセスで起動済みなら services のみ起動:

```bash
./env/bin/python std_t98_multi_service_launcher.py --services-only
```

子プロセスの標準出力をそのまま見たい場合:

```bash
./env/bin/python std_t98_multi_service_launcher.py --passthrough-output
```

起動コマンドだけ確認したい場合:

```bash
./env/bin/python std_t98_multi_service_launcher.py --dry-run
```

## IPC 既定値

- frame socket: `/tmp/std_t98_multi_frame.sock`
- voice socket: `/tmp/std_t98_multi_voice.sock`
- status socket: `/tmp/std_t98_multi_status.sock`

必要なら `STD_T98_MULTI_FRAME_SOCKET`、`STD_T98_MULTI_VOICE_SOCKET`、`STD_T98_MULTI_STATUS_SOCKET` で上書きできます。