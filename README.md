# std-t98-tools

ARIB STD-T98 のマルチチャネル受信系だけを残した GNU Radio / Python ツール群です。

## 現在のスコープ

このリポジトリは、`std_t98_multi_service_launcher.py` を起点にした 30ch 受信スタックだけを保持します。単一チャネル用エントリポイント、ファイル出力専用デコーダ、旧互換ラッパは削除済みです。

実行時の構成は次の 4 プロセスです。

- `std_t98_30ch_multi_rf_backend.py`
  - GNU Radio 側で 30ch の復調と同期語検出を行い、`FramePacket` を UDS へ送ります。
- `std_t98_multi_protocol_service.py`
  - `FramePacket` を復号し、RICH / SACCH / PICH を dashboard 向けに整形しつつ、TCH を `VoiceBurstPacket` として audio service に渡します。
- `std_t98_multi_secret_service.py`
  - repo 内の学習済みモデルを常駐ロードし、2450 ペイロード窓から秘話コード候補を探索します。鍵キャッシュはシステム全体で共有されます。
- `std_t98_multi_audio_service.py`
  - `VoiceBurstPacket` を受けて pyambelib で復号し、PCM を再生します。秘話中でも再生は止めず、裏で secret service へ十分長の 2450 データを渡し、鍵が確定した時点で以後のバーストへ反映します。

`std_t98_multi_service_launcher.py` はこれらをまとめて起動し、中央 dashboard と status 集約を担当します。

## 主要モジュール

- `std_t98_multi_service_launcher.py`
  - backend / protocol / secret / audio の起動順制御、Python 実行環境の自動選択、dashboard 更新。
- `std_t98_multi_sync.py`
  - GNU Radio Embedded Python block。各チャンネルの同期語相関を行い、`FramePacket` を送信。
- `core/protocol/`
  - デホワイトニング、フレーム分解、RICH / SACCH / PICH / TCH 処理。
- `core/audio/` と `core/crypto/`
  - AMBE 変換、秘話復号、音声再生用処理。
- `core/secret/`
  - 秘話推論モデル、キャッシュ検証、FFNN + hybrid の多段鍵探索。
- `ipc/message_schema.py`
  - `FramePacket`、`VoiceBurstPacket`、`StatusPacket`、`SecretCrackRequestPacket`、`SecretCrackResultPacket` のバイナリ schema。
- `ipc/transport/uds_seqpacket.py`
  - Unix Domain Socket の `SOCK_SEQPACKET` transport。
- `firdes.py`
  - バックエンドで使うフィルタタップ生成。

## 動作要件

- Linux
- GNU Radio / SoapySDR / RTL-SDR が使える Python 環境
- `pyambelib`、`sounddevice`、`torch`、`safetensors` が使える Python 環境

学習済みモデルはこの repo 内の `models/secret_voice/` に配置されます。実行時に外部プロジェクトのパスは参照しません。

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

音声再生は既定で PortAudio / PipeWire のデフォルト遅延を使います。低遅延に戻したい場合だけ、起動前に `STD_T98_AUDIO_LATENCY=low` を設定してください。低速 CPU では既定値のほうが安定します。

起動コマンドだけ確認したい場合:

```bash
./env/bin/python std_t98_multi_service_launcher.py --dry-run
```

## テスト

仮想環境に `pytest` が入っていれば、次で回帰テストを実行できます。

```bash
./env/bin/python -m pytest
```

## トラブルシュート

- 音声が止まったり起動直後に不安定になる場合は、まず `systemctl --user restart wireplumber pipewire pipewire-pulse` を試してください。
- N150 クラスの低消費電力 CPU では、`STD_T98_AUDIO_LATENCY=low` は安定性を落とすことがあります。既定値のまま使うほうが無難です。
- launcher の Python 環境チェックは import ごとにタイムアウトするので、音声スタック異常時でも起動全体が無限待機しにくくしています。

## IPC 既定値

- frame socket: `/tmp/std_t98_multi_frame.sock`
- voice socket: `/tmp/std_t98_multi_voice.sock`
- status socket: `/tmp/std_t98_multi_status.sock`
- secret request socket: `/tmp/std_t98_multi_secret_request.sock`
- secret result socket: `/tmp/std_t98_multi_secret_result.sock`

必要なら `STD_T98_MULTI_FRAME_SOCKET`、`STD_T98_MULTI_VOICE_SOCKET`、`STD_T98_MULTI_STATUS_SOCKET`、`STD_T98_MULTI_SECRET_REQUEST_SOCKET`、`STD_T98_MULTI_SECRET_RESULT_SOCKET` で上書きできます。