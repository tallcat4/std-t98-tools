# std-t98-tools

ARIB STD-T98 (デジタル簡易無線) の信号を受信・復号するための GNU Radio および Python ツール群です。

## 概要

本リポジトリには、RTL-SDR等のSDRを用いてデジタル簡易無線のRF信号を受信し、4FSKの復調、信号処理を行うバックエンド(GNU Radio)と、得られたシンボルからフレームごとの各種チャネル(RICH, SACCH, PICH, TCH等)をデインターリーブ・ビタビ復号して情報を取り出すフロントエンド(Pythonスクリプト)が含まれています。

AMBEフレームを含むTCHチャネルはファイルに出力され、後段のAMBEデコーダ等での音声復元に利用可能な形式で保存されます。(ThumbDVの3600x2450モードを用いれば、復調して再生できることを確認しています)

## ファイル構成

### 共通

- `firdes.py`
  - フィルタ設計用のスクリプトです。送受信用のRRCフィルタのタップ係数を生成し、各バックエンドから参照されます。

---

### シングルチャンネル系

1チャンネルのみを受信・復号するツール群です。

- `std_t98_rf_backend.grc`
  - GNU RadioによるRFフロントエンド処理のフローグラフ（GRC形式）。
  - SoapySDR経由で信号を受信、フィルタ処理、シンボル同期を実施します。
  - 同期ワードの相関検出を行い、フレーム単位で切り出した軟判定シンボルをZeroMQ (tcp://127.0.0.1:5555) 経由で送信します。


- `std_t98_frame_decoder.py`
  - ZeroMQ経由で軟判定シンボルを受け取り、各チャネルの復号を行います。
  - デホワイトニング、ビタビ復号、誤り訂正を行います。
  - 以下の各チャネルの抽出・復号・誤り訂正結果をコンソールに出力します。
    - **RICH** (Radio Information Channel): バースト種別、モード情報など。
    - **SACCH** (Slow Associated Control Channel): 呼び出し元情報、ユーザーコード、メーカーコードなど。
    - **PICH** (Parameter Information Channel): CSM情報など(同期バースト時)。
  - **TCH** (Traffic Channel) 通信用バーストの音声データは、`output.ambe`および`output.burst`ファイルとして出力されます。

- `std_t98_realtime_receiver.py`
  - `std_t98_frame_decoder.py` の音声リアルタイム再生版（シングルチャンネル）。
  - 1チャンネルのZMQストリームを受け取り、TCHデータをAMBE+2デコード後に即時PCM音声として再生します。
  - コンソールにはRXインジケーター、CSM(PICH)、SACCH情報をインプレース更新で表示します。
  - `DECRYPTION_KEY` 変数でLFSR初期状態（0〜32767）を指定することで秘話復号が可能です。

---

### マルチチャンネル系

最大30チャンネルを同時に受信・復号するツール群です。PFBチャンネライザを用いて1台のSDRで全チャンネルを並列処理します。

- `std_t98_30ch_multi_rf_backend.py`
  - 30チャンネル同時受信のためのPythonスクリプト版RFバックエンド。
  - RTL-SDR を中心周波数 351.29375 MHz / サンプルレート 1.2 MHz で動作させ、Stage 1 デシメーション（÷4）→ PFBチャンネライザ（48ch分割）→ 各チャンネルごとの Stage 2 リサンプリング・FM復調・シンボル同期という3段階の処理パイプラインで30チャンネルを並列処理します。
  - PFBチャンネルマップは、CH1〜CH15を Bin 33〜47、CH16 を Bin 0、CH17〜CH30 を Bin 1〜14 に割り当てています。
  - 各チャンネルの復調済みシンボルは `std_t98_multi_sync.sync_word_correlator` に渡されます。

- `std_t98_multi_sync.py`
  - マルチチャンネル対応の同期ワード相関器（GNU Radio Embedded Pythonブロック）。
  - 複数チャンネル分の入力ポート (`num_channels` 個の `float32`) を持つ `gr.sync_block` として動作します。
  - 各チャンネルで独立したシフトレジスタと収集バッファを管理し、同期ワードとのSSE（Sum of Squared Errors）が閾値以下になった時点でパケット収集を開始します。
  - 収集完了後、チャンネル番号（4バイト、リトルエンディアン）＋192シンボルの軟判定データをZeroMQ PUSHソケット (tcp://127.0.0.1:5555) 経由で送信します。

- `std_t98_multi_frame_decoder .py`
  - マルチチャンネル対応のフレームデコーダ（ファイル出力）。
  - ZMQからチャンネル番号付きパケットを受け取り、RICH / SACCH / PICH / TCH の復号結果をコンソールに詳細表示します。
  - 通信用バースト受信時は、チャンネルごとに `output_ch<N>.ambe` および `output_ch<N>.burst` ファイルを生成してAMBE+2音声データを書き出します。

- `std_t98_multi_realtime_receiver.py`
  - マルチチャンネル対応のリアルタイム音声受信・再生スクリプト。
  - 各チャンネルの `ChannelContext` オブジェクトで受信状態（OPEN/CLOSE）、CSM、SACCHを独立管理します。
  - 通信用バースト受信時は AMBE+2 デコードを行い、PipeWire経由でリアルタイムにPCM音声を再生します（8kHz → 48kHz アップサンプリング）。
  - コンソールには全受信チャンネルのダッシュボードをインプレース更新で表示します。
  - `DECRYPTION_KEY` 変数でLFSR初期状態（0〜32767）を指定することで秘話復号が可能です。

## 動作環境

* GNU Radio 3.10系
* Python 3.8系
* RTL-SDR v4

## 使い方

### シングルチャンネル受信（ファイル出力）

1. GNU Radio フローグラフ (`std_t98_rf_backend.grc`) を起動し、サンプリングとZeroMQへのパケット送信を開始します。
2. 同時に `std_t98_frame_decoder.py` を実行し、復号情報とAMBEパケットの抽出を待ち受けます。
3. ターミナルに復調された各チャネルの情報が出力されます。
4. 抽出された `output.ambe` もしくは `output.burst` をThumbDVやmbelibなどでWAV音声などに変換することも可能です。

```bash
python3 std_t98_rf_backend.py &   # または GRC GUI から起動
python3 std_t98_frame_decoder.py
```

### シングルチャンネル受信（リアルタイム音声再生）

1. `std_t98_rf_backend.grc` または `std_t98_rf_backend.py` を起動します。
2. `std_t98_realtime_receiver.py` を実行します。

```bash
python3 std_t98_rf_backend.py &
python3 std_t98_realtime_receiver.py
```

秘話を受信する場合は `std_t98_realtime_receiver.py` 内の `DECRYPTION_KEY` を送信機と同じ値（0〜32767）に設定してください。

### マルチチャンネル受信（最大30ch同時・ファイル出力）

1. `std_t98_30ch_multi_rf_backend.py` を起動します（内部で `std_t98_multi_sync.py` が使用されます）。
2. `std_t98_multi_frame_decoder .py` を実行します。チャンネルごとに `output_ch<N>.ambe` と `output_ch<N>.burst` が生成されます。

```bash
python3 std_t98_30ch_multi_rf_backend.py &
python3 std_t98_multi_frame_decoder.py
```

### マルチチャンネル受信（最大30ch同時・リアルタイム音声再生）

1. `std_t98_30ch_multi_rf_backend.py` を起動します。
2. `std_t98_multi_realtime_receiver.py` を実行します。

```bash
python3 std_t98_30ch_multi_rf_backend.py &
python3 std_t98_multi_realtime_receiver.py
```

---

### 出力ファイルについて

`output.ambe` / `output_ch<N>.ambe` は、AMBE+2音声データのみを順番にバイナリに書き出したものです。`output.burst` / `output_ch<N>.burst` はこれに加えて、受信データにおけるフレームの開始位置を示すマーカー（`0xFF`）を付与したものです。

STD-T98では、一フレームに4つのAMBE+2音声データが含まれています。秘話処理はフレーム単位で行われるため、どこがフレームの開始位置なのかを把握する必要があります。
秘話の解読処理を行う場合、`.ambe` ではなく `.burst` を用いて、フレームの開始位置を示す `0xFF` マーカーから4つの音声データを順番に取得する必要があります。

`std_t98_30ch_multi_rf_backend.py` の中心周波数・サンプルレート等は、スクリプト内冒頭の定数を編集することで変更できます。