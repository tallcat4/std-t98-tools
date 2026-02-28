# std-t98-tools

ARIB STD-T98 (デジタル簡易無線) の信号を受信・復号するための GNU Radio および Python ツール群です。

## 概要

本リポジトリには、RTL-SDR等のSDRレシーバを用いてデジタル簡易無線のRF信号を受信し、4FSKの復調、信号処理を行うバックエンド(GNU Radio)と、得られたシンボルからフレームごとの各種チャネル(RICH, SACCH, PICH, TCH等)をデインターリーブ・ビタビ復号して情報を取り出すフロントエンド(Pythonスクリプト)が含まれています。

AMBEフレームを含むTCHチャネルはファイルに出力され、後段のAMBEデコーダ等での音声復元に利用可能な形式で保存されます。(ThumbDVの3600x2450モードを用いれば、復調して再生できることを確認しています)

## ファイル構成

- `std_t98_rf_backend.grc`
  - GNU RadioによるRFフロントエンド処理ブロック。
  - SoapySDR経由で信号を受信、フィルタ処理、シンボル同期を実施します。
  - 同期ワードの相関検出を行い、フレーム単位で切り出した軟判定シンボルをZeroMQ (tcp://127.0.0.1:5555) 経由で送信します。

- `std_t98_frame_decoder.py`
  - ZeroMQ経由で軟判定シンボルを受け取り、各チャネルの復号を行います。
  - デホワイトニング、ビタビ復号、誤り訂正を行います。
  - 以下の各チャネルの抽出・復号・誤り訂正結果をコンソールに出力します。
    - **RICH** (Radio Information Channel): バースト種別、モード情報など。
    - **SACCH** (Slow Associated Control Channel): 呼び出し元情報、ユーザーコード、メーカーコードなど。
    - **PICH** (Parameter Information Channel): CSM情報など(同期バースト時)。
  - **TCH** (Traffic Channel) 通信用バーストの音声データは、`output.ambe` として出力されます。

- `firdes.py`
  - フィルタ設計用のスクリプトです。送受信用のフィルタのタップ係数を生成します。

## 動作環境

* GNU Radio 3.10系
* Python 3.8系
* RTL-SDR v4

## 使い方

1. GNU Radio フローグラフ (`std_t98_rf_backend.grc`) を起動し、サンプリングとZeroMQへのパケット送信を開始します。
2. 同時に `std_t98_frame_decoder.py` を実行し、復号情報とAMBEパケットの抽出を待ち受けます。
3. ターミナルに復調された各チャネルの情報が出力されます。
4. 抽出された `output.ambe` をThumvDVやmbelibなどでWAV音声などに変換することも可能です。

チャネル1を復調するためのパラメータが設定済みです。必要に応じてGNURadio Companionでパラメータを変更して使用してください。