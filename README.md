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
  - SoapySDR 経由で SDR を駆動し、PFB チャンネライザを使って 30 チャンネルを並列に処理する RF バックエンドです。既定では RTL-SDR / 中心周波数 351.29375 MHz / サンプルレート 1.2 MHz で動作しますが、これらは設定ファイルおよび CLI で変更できます（後述の「SDR 設定」参照）。
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

### tools/

実機立ち上げと不具合切り分けのための診断ツールです。復号パイプラインには含まれません。

- `tools/std_t98_channel_scope.py`
  - 1 チャンネルを GUI で見ます。RF スペクトラム、ウォーターフォール、チャンネルスペクトラム、アイパターン、復調シンボルの 5 面。SDR を開く処理はバックエンドと共有するので、見えているものはバックエンドが見ているものです。`--replay` で録音を再生できます。
- `tools/std_t98_record_iq.py`
  - チャンネライザ入力（30ch 全部）をファイルに録ります。1 回の送信をオフラインで何度でも解析できます。
- `tools/std_t98_analyse_capture.py`
  - 録音を測ります。送信のあったチャンネルと時間帯、周波数オフセット、実シンボルレート、同期語の検出数を 1 コマンドで出します。

### 補助データ

- `models/secret_voice/`
  - 秘話鍵探索に使う学習済みモデルです。
- `firdes.py`
  - バックエンドで使うフィルタタップ生成コードです。
- `tests/`
  - packet schema、launcher、audio service、UDS transport などの回帰テストです。

## 動作要件

Linux 専用です（IPC に `AF_UNIX` の `SOCK_SEQPACKET` を使います）。

依存は **RF 系** と **音声系** に分かれており、両方を 1 つの Python 環境に入れる必要はありません。launcher が backend 用 Python と service 用 Python を別々に解決するのは、この分割を前提にしているためです。既定では service 側に `env/bin/python`、backend 側にシステム Python を優先します。

| 区分 | 対象プロセス | 依存 | ファイル |
| --- | --- | --- | --- |
| RF 系 | RF backend / protocol service | GNU Radio（`gnuradio.soapy` 込み）、SoapySDR とデバイス別モジュール、`numpy` | `requirements-rf.txt` |
| 音声系 | audio service / secret service | `sounddevice`（+ PortAudio）、`pyambelib`、`torch`、`safetensors`、`numpy` | `requirements-audio.txt` |
| 開発 | テスト | `pytest` | `requirements-dev.txt` |

GNU Radio と SoapySDR は PyPI からは入りません。ディストリのパッケージか radioconda を使ってください。

```bash
# Debian / Ubuntu の例（使う SDR の module だけ入れれば十分です）
sudo apt install gnuradio libsoapysdr0.8 soapysdr-tools \
    soapysdr-module-rtlsdr soapysdr-module-uhd soapysdr-module-hackrf
pip install -r requirements-rf.txt

# 音声系（別環境でも可）
sudo apt install libportaudio2
pip install -r requirements-audio.txt
```

`pyambelib` は PyPI に無いため `requirements-audio.txt` には含めていません。C コンパイラと Python ヘッダだけでビルドできます（mbelib-neo の C ソースを同梱しているので外部ライブラリは不要です）。

```bash
git clone https://github.com/tallcat4/pyambelib
./env/bin/pip install ./pyambelib
```

未導入でも音声復号以外は動作します。

`rich` が入っていると launcher dashboard は安定した live 描画を使います。未導入時は簡易表示へフォールバックします。

導入後の確認:

```bash
python3 -c "from gnuradio import gr, soapy; print(gr.version())"
SoapySDRUtil --find          # 接続中の SDR とそのデバイス引数を表示
```

### 動作確認済みバージョン

以下の組み合わせで RF バックエンドのフローグラフ構築まで確認しています。

| 項目 | バージョン |
| --- | --- |
| OS | Ubuntu 24.04 (Linux 7.0) |
| Python | 3.12.3 |
| GNU Radio | 3.10.9.2 |
| SoapySDR | 0.8.1（API 0.8.0） |
| numpy | 1.26.4 |

学習済みモデルは `models/secret_voice/` に同梱されています。実行時に外部プロジェクトのパスを参照する前提にはしていません。

## SDR 設定

RF バックエンドは SoapySDR 経由で SDR を駆動するため、RTL-SDR に限らず、対応ドライバがあれば USRP (`uhd`) / HackRF (`hackrf`) / Airspy などにも切り替えられます。SDR 固有の値（ドライバ、サンプルレート、周波数、ゲイン、周波数誤差校正など）はソースにベタ書きせず、TOML 設定ファイルと CLI 引数で与えます。

- 既定値は従来の RTL-SDR / 1.2 MHz の挙動を完全に再現します。何も指定しなければ従来どおり動きます。
- 設定の優先順位は「組み込み既定値 → 設定ファイル → CLI 引数」で、後のものが前を上書きします。
- チャンネライザのレート（初段リサンプラ比・PFB 段数由来の中間レート）は、サンプルレートから自動導出します。STD-T98 の 6.25 kHz ラスタを保つよう計算されるため、`sample_rate` を変えても PFB ビン幅は 6.25 kHz に保たれます。

設定例は `config.example.toml` にあります。コピーして使ってください。

```bash
# 設定ファイルで起動
python std_t98_30ch_multi_rf_backend.py --config config.toml

# 環境変数でパスを渡す
STD_T98_BACKEND_CONFIG=config.toml python std_t98_30ch_multi_rf_backend.py

# CLI で個別に上書き（USRP を 2 MHz、AGC 無効、ゲイン 40、全体ゲインを使用）
python std_t98_30ch_multi_rf_backend.py \
    --driver uhd --sample-rate 2000000 --no-agc --gain 40 --gain-element ""

# 実機を開かずに、解決後の設定と導出レートだけ確認する
python std_t98_30ch_multi_rf_backend.py --config config.toml --dry-run
```

主な CLI 引数: `--config` / `--driver` / `--device-args` / `--stream-args` / `--sample-rate` / `--freq` / `--gain` / `--gain-element` / `--antenna` / `--bandwidth` / `--agc` / `--no-agc` / `--bias-tee` / `--no-bias-tee` / `--freq-correction` / `--pfb-channels` / `--dry-run`。

### デバイスの指定

`--driver` はドライバ種別を選ぶだけなので、同型の SDR が複数繋がっている場合や、ドライバに機種を教える必要がある場合は `--device-args` を併用します。内容は SoapySDR のデバイス文字列に `driver=<driver>,<device_args>` の形でそのまま連結されます。接続されているデバイスとその引数は `SoapySDRUtil --find` で確認できます。

```bash
# シリアルで 1 台を指定
python std_t98_30ch_multi_rf_backend.py --driver rtlsdr --device-args "serial=00000001"

# USRP の機種を明示
python std_t98_30ch_multi_rf_backend.py --driver uhd --device-args "type=b200"
```

### ストリーム引数

`stream_args` はドライバごとに既定値が変わります。未指定の場合、RTL-SDR には従来どおり `bufflen=16384` が渡り、それ以外のドライバには何も渡しません。SoapySDR はデバイスが公開していないストリーム引数を受け取ると `Unsupported stream argument` でソース生成そのものに失敗するため、RTL-SDR 用の `bufflen` を USRP や HackRF に渡してはいけません。この既定のおかげで `--driver` を差し替えるだけで別の SDR に切り替えられます。

明示指定したい場合のみ `--stream-args` を使い、何も渡したくない場合は空文字 `""` を指定します。

### サンプルレート

対応レートはデバイスによって大きく異なります。既定の 1.2MHz は RTL-SDR では使えますが、USRP B210 系は離散的なレートしか持たず 1.2MHz を含みません。非対応のレートを指定した場合は起動時に失敗し、**そのデバイスで使える近いレート**を提示します。

```
ValueError: This device cannot sample at 1200000 Hz.
Nearest rates this device supports: 1230769, 1142857, 1066667, 1333333, 1000000
```

デバイスの公称値は `1230769.230769...` のような端数を持つことがありますが、丸めた値を指定しても自動的に公称値へスナップするので、上の表示をそのまま渡せます。

レートを変えてもチャンネライザは 6.25kHz ラスタを保つよう初段リサンプラ比を計算し直します。この比は分母を制限した有理近似なので厳密には誤差が出ますが、実測では最悪でも 0.1Hz 程度（6250Hz のビン幅に対して）です。実際の誤差は `--dry-run` の `bin_width_error` で確認できます。

設定したレートは適用後に読み戻して検証します。黙って別のレートに丸めるドライバがあると、全チャネルが同調を外すためです。

### アナログ帯域幅

`--bandwidth` はフロントエンドのアナログフィルタ幅です。未指定の場合、RTL-SDR 以外のドライバでは **サンプルレートと同じ値**を設定します。RTL-SDR はレートに追従して自前で設定するため触りません。

これは実害のある既定です。USRP B210 は何も指定しないと **56MHz 全開**のままで、2MHz でサンプリングしていても ±28MHz の信号がすべて折り返して混入します。`0` を指定するとデバイス任せになります。

### 周波数誤差の補正

`freq_err_offset` は**特定の 1 台の無線機の実測校正値**であって、ドライバの性質ではありません。そのため測定した機種にのみ適用され（rtlsdr: -340 Hz）、**それ以外のデバイスは 0 から始まります**。

他機の値を流用してはいけません。復調器は周波数誤差をシンボル列の DC レベルに変換し、同期検出は DC 除去なしの絶対レベル比較なので、**-340 Hz だけで 1.24 シンボル単位の DC = SSE 15.4 となり、しきい値 14.8 を単独で超えます**。

自機の値は `--freq-err-offset` で与えます。`--dry-run` の `[sdr.resolved]` に、解決後の補正値と実際の同調周波数が出ます。

```bash
python3 std_t98_30ch_multi_rf_backend.py --driver uhd --freq-err-offset 1030
```

測り方は `tools/std_t98_channel_scope.py` が使えます。復調シンボルの中心が 0 からずれていれば、そのズレ量 × 273.9 Hz が残差です。

### スケルチ

`--squelch` はチャンネルごとのスケルチ閾値（dB、既定 -25）です。**絶対レベルなので、SDR のゲインとスケーリングに依存します。**

同じ電界強度でも、ある機種では -25 dB に、別の機種では -40 dB になります。閾値が高すぎると**全チャンネルが無音化され、しかも何の表示も出ません** — 周波数補正を誤ったときと同じ、「正常に動いているのに何も復号しない」症状になります。実測では B210 でゲイン 10 のとき -39.9 dB で、既定の -25 dB では同期検出が 0、-60 dB にすると 240 回検出しました。

新しい SDR では `std_t98_analyse_capture.py` でチャンネル電力を測り、それより十分低い値を設定してください。ゲインを上げて既定を満たす手もありますが、電力は送信距離や条件でも変わるため、閾値側に余裕を持たせる方が確実です。

### アンテナポート

RX ポートが複数ある機種では `--antenna` でどの端子から受けるかを選びます。未指定ならドライバ既定のままにするので、入力が 1 つしかない RTL-SDR では何も起きません。USRP B210 系は `TX/RX` と `RX2` を持ち、既定は `RX2` です（基板上の `RXA` 端子に対応。`TX/RX` は `TRXA`）。**アンテナを挿した端子と選択が一致していないと、何も受信できないのに正常動作しているように見えます。**

存在しない名前を指定した場合は起動時に即座に失敗し、利用可能な名前を表示します。

```bash
python std_t98_30ch_multi_rf_backend.py --driver uhd --antenna "TX/RX"
```

### デバイス依存機能の扱い

ゲイン要素名（`gain_element`）は RTL-SDR では `TUNER` が既定ですが、デバイスにその要素が無い場合や空文字を指定した場合はデバイス全体のゲインを設定します。bias tee や周波数補正 (ppm) は、デバイスが対応している場合のみ適用され、非対応でもエラーにはなりません。

`--dry-run` は解決後のデバイス文字列とストリーム引数を `[sdr.resolved]` として表示するため、実機を開く前に「実際に SoapySDR へ何が渡るか」を確認できます。

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

## 音声スタックなしで受信を確認する

新しい SDR や新しい環境に移したとき、まず確かめたいのは「RF 段が正しく受信できているか」です。これは **音声系の依存（`pyambelib` / `torch` / `sounddevice`）を一切入れずに**確認できます。protocol service は numpy だけで動き、voice socket は自分で bind するため、audio service が居なくても待ち続けたりはしません。

手順は 2 プロセスです。backend が frame socket を bind し、protocol service がそこへ接続するので、**起動順は backend が先**です。

```bash
# 1) 実機を開く前に、解決後の設定を確認する
python3 std_t98_30ch_multi_rf_backend.py --dry-run --driver uhd --sample-rate 2000000     --gain-element PGA --antenna RX2

# 2) RF backend（この端末は開いたままにする）
python3 std_t98_30ch_multi_rf_backend.py --driver uhd --sample-rate 2000000     --gain-element PGA --no-agc --gain 40 --antenna RX2

# 3) 別端末で protocol service
python3 std_t98_multi_protocol_service.py
```

正常なら protocol service 側に dashboard が出ます。無信号のときは次の表示になります。

```
[STD-T98 Multi Protocol Service]
  Waiting for signals...
```

電波を受けると各チャンネルの行が現れ、sync / frame / SACCH の成否が更新されます。`--headless` を付けると dashboard を止めて、`--status-socket` 経由の指標だけを流せます。

### 1 チャンネルを目で見る

数値だけで切り分けられないときは、GUI で受信チェーンの各段を直接見られます。

```bash
python3 tools/std_t98_channel_scope.py --driver uhd --sample-rate 2000000 \
    --antenna "TX/RX" --gain 10 --gain-element PGA --no-agc --channel 0 --no-squelch
```

`--channel` は内部インデックス（0〜29）で、**0 が登録局 ch1 = 351.20000 MHz** です。ウィンドウ表題と標準出力に両方の番号と実周波数を出します。

4 つの表示がチェーンの順に並びます。

| 表示 | 分かること |
| --- | --- |
| RF スペクトラム + ウォーターフォール | 電波が出ているか、30ch のどこか |
| チャンネルスペクトラム | 搬送波が 6.25kHz の枠の中心にあるか、ズレているか |
| アイパターン | 判定できる開口があるか |
| 復調シンボル | シンボル同期が引き込んだか |

DSP はバックエンドと同一の設定（`fsk_dev` 315、RRC ロールオフ 0.2、Gardner TED など）を使い、SDR を開く処理も `core/rf/soapy_source.py` を共有します。レート検証もアンテナ確認も帯域幅もバックエンドと同じなので、**画面に出るものがバックエンドの見ているもの**です。

`--no-squelch` はスケルチ（-25dB）を迂回します。弱い信号だとスケルチで消えてアイパターンが平坦になるため、切り分け中は付けておくのが安全です。

### 録って解析する

GUI を睨むより、1 回録って測る方が速く確実です。送信のたびに人を待たせる必要もありません。

```bash
# 30秒録る（>>> RECORDING <<< が出てから送信）
python3 tools/std_t98_record_iq.py --driver uhd --sample-rate 2000000 \
    --antenna "TX/RX" --gain-element PGA --no-agc --gain 10 \
    --freq-err-offset 1030 --seconds 30 -o capture.cf32

# 測る
python3 tools/std_t98_analyse_capture.py capture.cf32

# 同じ録音を GUI で見る（SDR 不要）
python3 tools/std_t98_channel_scope.py --replay capture.cf32 --channel 0 --no-squelch
```

解析の出力例です。

```
noise floor -40.7 dB
  ch0  (登録局 ch1 )  48.1 dB above floor, active  1.30-25.40s
--- ch0 (登録局 ch1), 1.3-25.4s ---
frequency offset        +4 Hz (DC +0.02 on the symbol stream)
                  sync tolerates about 333 Hz
symbol rate       2400.0311 baud (+13.0 ppm), phase drift +0.810 samples/s
sync detections   267 (best SSE 0.25, threshold 14.8)
```

### 録音でフルスタックを通す

バックエンドの `--replay` を使えば、**SDR も送信も無しで 4 プロセス全体を検証**できます。移植先での受け入れテストとして使ってください。

```bash
# 3サービスを先に起動（audio は secret に接続するので secret も必要）
./env/bin/python std_t98_multi_protocol_service.py --headless &
./env/bin/python std_t98_multi_secret_service.py --headless &
./env/bin/python std_t98_multi_audio_service.py &

# 録音を再生
python3 std_t98_30ch_multi_rf_backend.py --replay capture.cf32 --squelch -40
```

launcher から一括で起動することもできます。`--backend-arg` でバックエンドに引数を渡せるので、録音再生のフルスタックが 1 コマンドで立ち上がります。

```bash
python3 std_t98_multi_service_launcher.py \
    --backend-arg=--replay --backend-arg=capture.cf32 \
    --backend-arg=--squelch --backend-arg=-40
```

実機でも同じ形で起動できますが、引数が増えるので設定ファイルの方が扱いやすいです。`STD_T98_BACKEND_CONFIG` は launcher の子プロセスにも継承されます。

```bash
# ~/b210.toml に SDR 設定と [demod] squelch_threshold を書いておく
STD_T98_BACKEND_CONFIG=~/b210.toml python3 std_t98_multi_service_launcher.py
```

複数チャンネルで同時に送信があれば、それぞれの CH 行が独立に OPEN になり、並列に復調・再生されます。

音声が再生されれば、RF 段から AMBE 復号・音声出力までの全経路が通っています。暗号化された送信では、secret service が鍵を推定してキャッシュに登録し（dashboard の Secret Cache に鍵が現れます）、以降その鍵で復号されます。

**起動順に注意してください。** audio service は voice / secret_request / secret_result の 3 ソケットに順に接続し、いずれも無限リトライします。secret service を起動していないと voice 接続後にブロックし、メインループに入りません（protocol の dashboard には `Traffic (no client)` と出ます）。

### 新しい SDR での周波数校正

`freq_err_offset` の値はこの手順で決めます。

1. `--freq-err-offset 0` で 30 秒録音し、その間に送信する
2. `std_t98_analyse_capture.py` の `frequency offset` を読む
3. その値をそのまま `--freq-err-offset` に渡して録り直す
4. `frequency offset` が数十 Hz 以内、`sync detections` が 0 でなくなれば完了

許容範囲は約 ±333 Hz しかないので、目分量では合いません。この値には SDR 側と送信機側の誤差が両方含まれるため、送信機を変えたら測り直しになります。

うまくいかないときの切り分け順:

1. `SoapySDRUtil --find` — デバイスが見えるか
2. `--dry-run` の `[sdr.resolved]` — device string・stream args・antenna・bandwidth が意図どおりか
3. backend が例外なく走り続けるか（レート非対応なら起動時に候補付きで失敗します）
4. **アンテナを挿した端子と `--antenna` が一致しているか** — ここがズレていると、エラーも警告も出ないまま永久に受信しません
5. それでも `sync=0` なら、単にその時間帯に電波が出ていない可能性があります

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

`env/` は launcher が service 用 Python として探すパスなので、テスト用の環境は `.venv/` に分けます。GNU Radio は distro パッケージなので `--system-site-packages` が要ります。

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

`tests/test_multi_audio_service.py` だけは音声系依存（`sounddevice`）を必要とします。RF 系のみの環境では `--ignore=tests/test_multi_audio_service.py` を付けてください。GNU Radio にも依存しないテストは `python3 -m unittest tests.test_backend_config` だけでも実行できます。

音声スタックまで揃った環境なら `./env/bin/python -m pytest` でも同じものが回ります。

## トラブルシューティング

- 音声が止まったり起動直後に不安定になる場合は、まず `systemctl --user restart wireplumber pipewire pipewire-pulse` を試してください。
- N150 クラスの低消費電力 CPU では、`STD_T98_AUDIO_LATENCY=low` は安定性を落とすことがあります。
- launcher の Python 環境チェックは import ごとにタイムアウトするので、音声スタック異常時でも起動全体が無限待機しにくくしています。

## IPC 既定値

ソケットは `XDG_RUNTIME_DIR` が設定されていればその下の `std-t98/` に、無ければ `/tmp` に置かれます。

- frame socket: `$XDG_RUNTIME_DIR/std-t98/std_t98_multi_frame.sock`
- voice socket: `$XDG_RUNTIME_DIR/std-t98/std_t98_multi_voice.sock`
- status socket: `$XDG_RUNTIME_DIR/std-t98/std_t98_multi_status.sock`
- secret request socket: `$XDG_RUNTIME_DIR/std-t98/std_t98_multi_secret_request.sock`
- secret result socket: `$XDG_RUNTIME_DIR/std-t98/std_t98_multi_secret_result.sock`

`XDG_RUNTIME_DIR`（通常 `/run/user/$UID`、パーミッション 0700）はユーザ単位で分離され、ログアウト時に自動で片付けられます。`/tmp` は全ユーザ共有かつ誰でも書けるため、同じマシンで複数ユーザが起動するとパスが衝突し、異常終了で残ったソケットも放置されます。ディレクトリは最初に bind したプロセスが 0700 で作成します。

置き場所は `STD_T98_RUNTIME_DIR` でまとめて変更でき、個別のパスは `STD_T98_MULTI_FRAME_SOCKET`、`STD_T98_MULTI_VOICE_SOCKET`、`STD_T98_MULTI_STATUS_SOCKET`、`STD_T98_MULTI_SECRET_REQUEST_SOCKET`、`STD_T98_MULTI_SECRET_RESULT_SOCKET` で上書きできます（個別指定が優先）。

launcher は子プロセスに環境変数をそのまま引き継ぐため、launcher 経由なら全プロセスが同じパスを使います。backend と service を別々に起動する場合は、双方の `XDG_RUNTIME_DIR` が一致していることを確認してください（例: systemd unit や `sudo` 経由だと未設定になり `/tmp` 側にずれます）。