# std-t98-tools

ARIB STD-T98 (デジタル簡易無線) の信号をマルチチャネルで受信・復号し、音声再生まで行う GNU Radio / Python ツール群です。

## 概要

本リポジトリには、RTL-SDR 等の SDR を用いて 351MHz 帯の STD-T98 信号を受信し、30 チャンネルを並列に復調・同期・フレーム復号し、最終的に TCH 音声をリアルタイム再生するための一連のコンポーネントが含まれています。

処理系は大きく次の 4 プロセスで構成されています。

1. `std_t98_30ch_multi_rf_backend.py`
   - GNU Radio 側で 30ch の復調、シンボル同期、同期語検出を行い、`FramePacket` を生成します。
2. `std_t98_multi_protocol_service.py`
   - `FramePacket` を受け取り、RICH / SACCH / PICH / TCH を解析し、音声に相当するバーストを `VoiceBurstPacket` として後段へ渡します。
3. `std_t98_multi_audio_service.py`
   - `VoiceBurstPacket` を AMBE 復号して PCM に変換し、チャネルごとに短時間バッファしたうえでミックス再生します。
4. `std_t98_multi_secret_service.py`
   - 秘話音声の鍵探索を担当し、2450 ペイロード窓から学習済みモデルで鍵候補を推定します。

これらをまとめて起動し、状態監視用 dashboard と status 集約を行うのが `std_t98_multi_service_launcher.py` です。

## 処理構成

受信から再生までのデータフローは次のようになります。

```text
RTL-SDR
  -> std_t98_30ch_multi_rf_backend.py
  -> std_t98_multi_sync.py / core.rf.sync_word_correlator
  -> FramePacket (UDS)
  -> std_t98_multi_protocol_service.py
  -> VoiceBurstPacket (UDS)
  -> std_t98_multi_audio_service.py
  -> PCM / PortAudio / PipeWire

                +-> SecretCrackRequestPacket (UDS)
                +-> std_t98_multi_secret_service.py
                +-> SecretCrackResultPacket (UDS)
```

launcher は別経路で各プロセスの `StatusPacket` を集約し、チャンネル状態やデバッグ用メトリクスを表示します。

## ファイル構成

### 起動・サービス

- `std_t98_multi_service_launcher.py`
  - backend / protocol / audio / secret の起動順制御、Python 実行環境の解決、status 集約、dashboard 表示を担当します。
- `std_t98_30ch_multi_rf_backend.py`
  - RTL-SDR を中心周波数 351.29375 MHz / サンプルレート 1.2 MHz で動作させ、PFB チャンネライザを使って 30 チャンネルを並列に処理する RF バックエンドです。
- `std_t98_multi_protocol_service.py`
  - `FramePacket` を受け取り、デホワイトニング、RICH / SACCH / PICH / TCH 解析を行うフロントエンドです。
- `std_t98_multi_audio_service.py`
  - `VoiceBurstPacket` を受け取り、AMBE 復号、秘話解除、PCM 出力、チャネル間ミックスを行います。
- `std_t98_multi_secret_service.py`
  - 学習済みモデルを常駐ロードし、秘話鍵の推定と共有キャッシュ管理を行います。
- `std_t98_multi_sync.py`
  - GNU Radio Embedded Python block のラッパです。実体は `core/rf/sync_word_correlator.py` にあります。

### core/

- `core/rf/`
  - 同期語相関、フレーム切り出しなど RF 後段の共通処理を持ちます。
- `core/protocol/`
  - デホワイトニング、フレーム分解、RICH / SACCH / PICH / TCH の各デコーダを持ちます。
- `core/audio/`
  - pyambelib との橋渡し、AMBE 変換、PCM 化に関わる処理を持ちます。
- `core/crypto/`
  - PN 系列生成や秘話解除処理を持ちます。
- `core/secret/`
  - 秘話鍵探索ロジックとモデル利用コードを持ちます。
- `core/pipeline/`
  - dashboard 表示や runtime status の共通コードを持ちます。

### ipc/

- `ipc/message_schema.py`
  - `FramePacket`、`VoiceBurstPacket`、`StatusPacket`、`SecretCrackRequestPacket`、`SecretCrackResultPacket` のバイナリ schema を定義します。
- `ipc/transport/uds_seqpacket.py`
  - Unix Domain Socket `SOCK_SEQPACKET` ベースの transport 実装です。

### 補助データ

- `models/secret_voice/`
  - 秘話鍵探索に使う学習済みモデルです。
- `firdes.py`
  - バックエンドで使うフィルタタップ生成コードです。
- `tests/`
  - packet schema、launcher、audio service、UDS transport などの回帰テストです。

## 動作要件

- Linux
- GNU Radio / SoapySDR / RTL-SDR が使える Python 環境
- `pyambelib`、`sounddevice`、`torch`、`safetensors` が使える Python 環境
- `rich` が入っていると launcher dashboard は安定した live 描画を使います。未導入時は簡易表示へフォールバックします。

学習済みモデルは `models/secret_voice/` に同梱されています。実行時に外部プロジェクトのパスを参照する前提にはしていません。

launcher は backend 用 Python と service 用 Python を別々に解決できます。既定では service 側に `env/bin/python`、backend 側にシステム Python を優先します。

## 使い方

### フルスタック起動

```bash
./env/bin/python std_t98_multi_service_launcher.py
```

### backend を別起動して services のみ動かす場合

```bash
./env/bin/python std_t98_multi_service_launcher.py --services-only
```

### 子プロセスの標準出力をそのまま見たい場合

```bash
./env/bin/python std_t98_multi_service_launcher.py --passthrough-output
```

### デバッグ用メトリクスを dashboard に表示する場合

```bash
./env/bin/python std_t98_multi_service_launcher.py --show-debug-metrics
```

`--show-debug-metrics --passthrough-output` を併用すると、launcher の集計表示と各 child process のログを同時に確認できます。

### 主な実行オプション

- 音声再生は既定で PortAudio / PipeWire のデフォルト遅延を使います。低遅延にしたい場合は起動前に `STD_T98_AUDIO_LATENCY=low` を設定してください。
- audio service はチャネルごとの PCM を短時間バッファしてからミックスします。起動直後の再生には約 120ms のプリバッファを入れています。
- 再生音量は既定で 4.0 倍の出力ゲインをかけます。必要に応じて `STD_T98_AUDIO_GAIN=2.0` や `STD_T98_AUDIO_GAIN=4.0` のように調整できます。

### 起動コマンドだけ確認したい場合

```bash
./env/bin/python std_t98_multi_service_launcher.py --dry-run
```

## 状態確認とデバッグ

launcher の dashboard には、各チャンネルの RX 状態、protocol / audio / secret の状態、SACCH 情報、秘話鍵状態が表示されます。

`--show-debug-metrics` を付けると、さらに次の集計値を表示できます。

- RF 側
  - 同期検出数、frame IPC 成功/失敗数、同期判定指標と閾値
- protocol 側
  - frame 受信数、sync / traffic バースト数、RICH 失敗数、SACCH 成功/失敗数、voice IPC 成功/失敗数
- audio 側
  - voice burst 受信数、PCM enqueue 数、空 decode 数、キュー滞留量、trim 回数、underflow 回数、stream reopen 回数

複数チャネル受信時の音切れ調査では、RF の同期は取れているか、protocol から audio への IPC が詰まっていないか、audio 出力側で underflow や trim が増えていないかを見ると切り分けしやすくなります。

## テスト

仮想環境に `pytest` が入っていれば、次で回帰テストを実行できます。

```bash
./env/bin/python -m pytest
```

## トラブルシューティング

- 音声が止まったり起動直後に不安定になる場合は、まず `systemctl --user restart wireplumber pipewire pipewire-pulse` を試してください。
- N150 クラスの低消費電力 CPU では、`STD_T98_AUDIO_LATENCY=low` は安定性を落とすことがあります。
- launcher の Python 環境チェックは import ごとにタイムアウトするので、音声スタック異常時でも起動全体が無限待機しにくくしています。

## IPC 既定値

- frame socket: `/tmp/std_t98_multi_frame.sock`
- voice socket: `/tmp/std_t98_multi_voice.sock`
- status socket: `/tmp/std_t98_multi_status.sock`
- secret request socket: `/tmp/std_t98_multi_secret_request.sock`
- secret result socket: `/tmp/std_t98_multi_secret_result.sock`

必要なら `STD_T98_MULTI_FRAME_SOCKET`、`STD_T98_MULTI_VOICE_SOCKET`、`STD_T98_MULTI_STATUS_SOCKET`、`STD_T98_MULTI_SECRET_REQUEST_SOCKET`、`STD_T98_MULTI_SECRET_RESULT_SOCKET` で上書きできます。