# std-t98-tools

ARIB STD-T98（デジタル簡易無線, 351 MHz 帯）の信号を SDR で受信し、30 チャンネルを並列に復調・復号して音声再生する GNU Radio / Python ツール群です。

- **マルチチャネル**: 30 チャンネルを同時に復調・監視・再生
- **マルチ SDR**: SoapySDR 対応デバイスなら RTL-SDR / USRP / HackRF / Airspy などを設定だけで切り替え
- **フルスタック**: RF 受信からプロトコル解析、AMBE 音声復号、音声出力まで
- **診断ツール同梱**: SDR なしでも録音から全経路を検証できる受け入れテスト

## 目次

- [仕組み](#仕組み)
- [動作環境](#動作環境)
- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [SDR 設定](#sdr-設定)
- [診断とトラブルシューティング](#診断とトラブルシューティング)
- [リファレンス](#リファレンス)
- [テスト](#テスト)
- [ライセンス](#ライセンス)

## 仕組み

受信から再生までを 4 つのプロセスに分け、Unix ドメインソケットで接続しています。まとめて起動・監視するのが launcher です。

```text
 SDR ──▶ RF backend ──FramePacket──▶ protocol ──VoiceBurstPacket──▶ audio ──▶ 音声出力
         (30ch 復調)                 (RICH/SACCH/                    (AMBE 復号・
                                      PICH/TCH 解析)                  ミックス再生)
                                          │                              │
                                          └──────────── secret ◀────────┘
                                             (秘話鍵の推定)
```

| プロセス | 役割 |
| --- | --- |
| `std_t98_30ch_multi_rf_backend.py` | SoapySDR で SDR を駆動し、PFB チャンネライザで 30ch を並列復調、シンボル同期・同期語検出まで行う |
| `std_t98_multi_protocol_service.py` | フレームをデホワイトニングし、RICH / SACCH / PICH / TCH を解析 |
| `std_t98_multi_audio_service.py` | 音声バーストを AMBE 復号し、PCM 化・チャネル間ミックス・再生 |
| `std_t98_multi_secret_service.py` | 学習済みモデルで秘話鍵を推定し、音声スクランブルを解除 |
| `std_t98_multi_service_launcher.py` | 上記の起動・状態集約・dashboard 表示 |

## 動作環境

- **Linux 専用**（IPC に `AF_UNIX` の `SOCK_SEQPACKET` を使用）
- Python 3.11 以降

依存は **RF 系** と **音声系** に分かれ、別々の Python 環境に入れられます。launcher が backend 用と service 用の Python を個別に解決するのはこのためで、既定では backend にシステム Python、service に `env/bin/python` を使います。

| 区分 | 対象 | 依存 |
| --- | --- | --- |
| RF 系 | backend / protocol | GNU Radio（`gnuradio.soapy` 込み）、SoapySDR + デバイスモジュール、numpy |
| 音声系 | audio | sounddevice（+ PortAudio）、pyambelib、numpy |
| 秘話系 | secret | torch、safetensors |
| 開発 | テスト | pytest |

秘話系（音声スクランブルの解除）は標準機能で、`setup.sh` が既定で導入します。torch を入れたくない用途向けに `--no-secret` で外すこともでき、その場合 audio service は secret service の不在を自動判定してクリア音声のみを復号・再生します（暗号化通信はスクランブルされたまま）。

動作を確認済みの組み合わせ:

| OS | Python | GNU Radio | SoapySDR |
| --- | --- | --- | --- |
| Ubuntu 24.04 | 3.12.3 | 3.10.9.2 | 0.8.1 (API 0.8.0) |

## インストール

GNU Radio と SoapySDR は PyPI に無いため、先にディストリのパッケージ（または [radioconda](https://github.com/ryanvolz/radioconda)）で導入します。SDR のモジュールは使う機種のものだけで十分です。

```bash
# Debian / Ubuntu の例
sudo apt install gnuradio libsoapysdr0.8 soapysdr-tools libportaudio2 \
    soapysdr-module-rtlsdr soapysdr-module-uhd soapysdr-module-hackrf
```

残りは `setup.sh` が用意します。service 用の `env/` を作り、音声系と `pyambelib`（ソースからビルド）を導入し、RF 系が使えるかを確認します。root は不要で、システム Python には触れず、再実行しても安全です。

```bash
./setup.sh                 # 全機能（秘話解読の torch を含む）
./setup.sh --no-secret     # 秘話解読 (torch) を省く
./setup.sh --with-dev      # 開発用に pytest も追加
```

<details>
<summary>手動で導入する場合</summary>

```bash
# RF 系（システム Python など、GNU Radio が見える環境へ）
pip install -r requirements-rf.txt

# 音声系
python3 -m venv --system-site-packages env
./env/bin/pip install -r requirements-audio.txt

# pyambelib は PyPI に無いのでソースからビルド（C コンパイラと Python ヘッダのみ、
# mbelib-neo の C ソースを同梱しているため外部ライブラリは不要）
git clone https://github.com/tallcat4/pyambelib
./env/bin/pip install ./pyambelib

# 秘話系
./env/bin/pip install -r requirements-secret.txt
```

`rich` が入っていると launcher の dashboard が安定した live 描画になります（未導入時は簡易表示）。学習済みモデルは `models/secret_voice/` に同梱済みで、外部パスは参照しません。
</details>

導入後、RF 環境と SDR の疎通を確認できます。

```bash
python3 -c "from gnuradio import gr, soapy; print(gr.version())"
SoapySDRUtil --find     # 接続中の SDR とデバイス引数を表示
```

## クイックスタート

launcher が全プロセスを起動します。SDR の設定は設定ファイルにまとめ、環境変数で渡すのが簡単です（子プロセスにも継承されます）。

```bash
# 設定を書く（例）: ~/std-t98.toml
#   [sdr]
#   driver = "uhd"
#   sample_rate = 2000000
#   antenna = "TX/RX"
#   [demod]
#   squelch_threshold = -40

STD_T98_BACKEND_CONFIG=~/std-t98.toml python3 std_t98_multi_service_launcher.py
```

RTL-SDR を既定設定で使う場合は、設定ファイルなしでそのまま起動できます。

```bash
python3 std_t98_multi_service_launcher.py
```

送信を受けると、該当チャンネルの行が `RX: [OPEN]` になり、復調・復号された音声が再生されます。複数チャンネルで同時に送信があれば、それぞれ独立に処理されます。停止は `Ctrl+C` です。

### デスクトップ GUI

launcher と同じスタックを、端末ではなく GUI で起動・監視できます。プロセスの起動・停止・状態集約は launcher と共通の `StackSupervisor` を使い、30 チャンネルをタイル表示します。

```bash
python3 -m app          # または ./std_t98_gui.py
```

PyQt5 が必要です（GNU Radio の Qt GUI に含まれるため、RF 環境が入っていれば追加インストールは不要）。ウィンドウの Start / Stop でスタックを起動・停止し、子プロセスが終了した場合は理由を表示します。

設定は折りたたみ式の Settings パネルで扱います（Start すると自動的に畳まれます）。

- **Config**: backend の TOML を選択（Browse）。`STD_T98_BACKEND_CONFIG` があれば初期値として拾い、以降は前回のパスを記憶します。選んだ設定は `--dry-run` と同じ内容（driver / レート / 同調周波数 / アンテナ / 帯域 など）をその場でプレビューし、ファイルが無ければ起動前に警告します。
- **Detect SDRs**: 接続中の SoapySDR デバイスを一覧し、設定の driver が実際に繋がっているかを確認します。
- **Backend**: `--replay` などの追加引数を渡せます（`--config` の後に付与）。

主な launcher オプション:

| オプション | 説明 |
| --- | --- |
| `--services-only` | backend を別途起動し、services だけを動かす |
| `--passthrough-output` | 各子プロセスのログを端末にそのまま出す |
| `--show-debug-metrics` | dashboard に集計メトリクスを表示 |
| `--backend-arg=...` | backend に引数を渡す（繰り返し可） |
| `--dry-run` | 実行せず、解決された起動コマンドだけ表示 |

音声再生の調整は環境変数で行います。`STD_T98_AUDIO_GAIN`（既定 4.0 倍）、`STD_T98_AUDIO_LATENCY=low`（低遅延、ただし低消費電力 CPU では不安定になることがあります）。

## SDR 設定

backend は SoapySDR 経由で SDR を駆動するため、対応ドライバがあれば機種を問わず切り替えられます。設定の優先順位は **組み込み既定値 → 設定ファイル（`--config` または `STD_T98_BACKEND_CONFIG`）→ CLI 引数** で、後のものが前を上書きします。既定値は RTL-SDR / 1.2 MHz の従来挙動を再現します。全項目のひな形は `config.example.toml` にあります。

チャンネライザのレートはサンプルレートから自動導出され、STD-T98 の 6.25 kHz ラスタを保つよう計算されます。`sample_rate` を変えても PFB ビン幅は 6.25 kHz に保たれます。

実機を開く前に、解決後の設定と導出レートを確認できます。

```bash
python3 std_t98_30ch_multi_rf_backend.py --config myradio.toml --dry-run
```

主な CLI 引数: `--driver` / `--device-args` / `--stream-args` / `--sample-rate` / `--freq` / `--gain` / `--gain-element` / `--antenna` / `--bandwidth` / `--agc` / `--no-agc` / `--freq-err-offset` / `--squelch` / `--config` / `--dry-run`。

### 新しい SDR で注意する項目

RTL-SDR 以外を使う際、機種によって調整が要る主な項目です。いずれも既定は RTL-SDR 向けで、他機では自動で無難な値に切り替わるか、設定が必要です。

- **アンテナ (`--antenna`)**: RX ポートが複数ある機種で受信端子を選びます。USRP B210 系は `TX/RX` と `RX2` を持ち、既定は `RX2`。**挿した端子と一致していないと、エラーも出ないまま何も受信しません。** 存在しない名前は起動時に候補付きで拒否されます。
- **サンプルレート (`--sample-rate`)**: 対応レートは機種依存です。非対応値を指定すると、そのデバイスで使える近いレートを提示して停止します。端数を持つ公称値（例 `1230769.23…`）には丸めた値からスナップします。
- **周波数誤差 (`--freq-err-offset`)**: 個体ごとの実測校正値で、機種をまたいで流用できません。ずれていると同期語を検出できません。許容は約 ±333 Hz。→ [周波数校正](#周波数校正)
- **スケルチ (`--squelch`)**: 絶対レベル（dB）のため、SDR のゲイン・スケーリングに依存します。高すぎると全チャンネルが無音化されます。ゲインより閾値側に余裕を持たせるのが確実です。
- **アナログ帯域幅 (`--bandwidth`)**: 未指定なら RTL-SDR 以外はサンプルレートに追従させます（RTL-SDR は自前で設定）。設定しないと折り返し混入の原因になります。
- **デバイス選択 (`--device-args`)**: 同型機が複数ある場合などに `serial=...` や `type=b200` を渡します。`driver=<driver>,<device_args>` として連結されます。
- **ストリーム引数 (`--stream-args`)**: RTL-SDR には既定で `bufflen=16384`、他機には何も渡しません（非対応の引数はソース生成に失敗するため）。

ゲイン要素名 (`gain_element`) は RTL-SDR で `TUNER`、無い機種や空文字ではデバイス全体のゲインになります。bias tee・周波数補正 (ppm) は対応機種でのみ適用され、非対応でもエラーにはなりません。

## 診断とトラブルシューティング

`tools/` に、実機立ち上げと不具合切り分けのための診断ツールがあります（復号パイプライン本体には含まれません）。

### 受信できているかを確認する

まず RF 段が受信できているかは、**音声系の依存なしで**確認できます。protocol service は numpy だけで動きます。

```bash
# 端末1: backend（先に起動して frame socket を bind）
python3 std_t98_30ch_multi_rf_backend.py --config myradio.toml

# 端末2: protocol service
python3 std_t98_multi_protocol_service.py
```

無信号なら `Waiting for signals...`、受信すると各チャンネルの行に sync / frame / SACCH の成否が出ます。

### 波形を目で見る

数値で切り分けられないときは、1 チャンネルを GUI で各段ごとに観察できます。RF スペクトラム、ウォーターフォール、チャンネルスペクトラム、アイパターン、復調シンボルの 5 面です。DSP と SDR 設定を backend と共有するため、表示される内容は backend が見ているものと一致します。

```bash
python3 tools/std_t98_channel_scope.py --config myradio.toml --channel 0 --no-squelch
```

`--channel` は内部インデックス（0–29、`0` が登録局 ch1 = 351.20000 MHz）。`--no-squelch` は切り分け中の弱信号を消さないために付けます。

### 録音して解析する

1 回録音すれば、送信のたびに待たずにオフラインで何度でも解析できます。

```bash
# 録音（>>> RECORDING <<< 表示後に送信）
python3 tools/std_t98_record_iq.py --config myradio.toml --seconds 30 -o capture.cf32

# 解析（アクティブなチャンネル・周波数オフセット・シンボルレート・同期検出数を表示）
python3 tools/std_t98_analyse_capture.py capture.cf32

# 録音を GUI で再生（SDR 不要）
python3 tools/std_t98_channel_scope.py --replay capture.cf32 --channel 0 --no-squelch
```

### SDR なしでフルスタックを検証する

backend の `--replay` は、録音をチャンネライザに流し込みます。SDR も送信もなしに全 4 プロセスを検証でき、別環境への移行時の受け入れテストになります。

```bash
python3 std_t98_multi_service_launcher.py \
    --backend-arg=--replay --backend-arg=capture.cf32 \
    --backend-arg=--squelch --backend-arg=-40
```

音声が再生されれば、RF 段から音声出力までの全経路が通っています。

### 周波数校正

`--freq-err-offset` の値は次の手順で決めます。許容は約 ±333 Hz と狭く、目分量では合いません（SDR 側と送信機側の誤差を両方含むため、送信機を変えたら測り直しです）。

1. `--freq-err-offset 0` で録音し、その間に送信する
2. `std_t98_analyse_capture.py` の `frequency offset` を読む
3. その値を `--freq-err-offset` に渡して録り直す
4. `frequency offset` が数十 Hz 以内、`sync detections` が 0 でなくなれば完了

### 切り分けの順序

受信できないときは次の順で確認します。

1. `SoapySDRUtil --find` — デバイスが見えるか
2. `--dry-run` の `[sdr.resolved]` — device string / stream args / antenna / bandwidth が意図どおりか
3. backend が例外なく走り続けるか（レート非対応なら起動時に候補付きで停止）
4. **アンテナ端子と `--antenna` が一致しているか**（不一致だと無警告で受信ゼロ）
5. スケルチが高すぎないか（`--no-squelch` で切り分け）
6. それでも `sync=0` なら、単にその時間帯に送信が無い可能性

### 音声の不具合

- 音が途切れる・起動直後に不安定: まず `systemctl --user restart wireplumber pipewire pipewire-pulse`
- `--show-debug-metrics --passthrough-output` を併用すると、集計表示と各プロセスのログを同時に確認できます。音切れ調査では、RF の同期、protocol→audio の IPC 詰まり、audio 出力の underflow / trim を順に見ると切り分けやすくなります。

## リファレンス

### ディレクトリ構成

```
std_t98_*.py            4 サービス + launcher
core/rf/                同期語相関、SDR 制御、レート導出
core/protocol/          デホワイトニング、RICH/SACCH/PICH/TCH デコーダ
core/audio/             pyambelib 連携、AMBE 変換、PCM 化
core/crypto/            PN 系列生成、秘話解除
core/secret/            秘話鍵探索とモデル利用
core/pipeline/          dashboard・runtime status 共通処理
ipc/                    バイナリ schema と UDS transport
tools/                  診断ツール（scope / record / analyse）
models/secret_voice/    秘話鍵探索の学習済みモデル
tests/                  回帰テスト
```

### IPC ソケット

ソケットは `XDG_RUNTIME_DIR`（通常 `/run/user/$UID`、0700、ユーザ単位で分離）配下の `std-t98/` に置かれます。未設定時は `/tmp` にフォールバックします。ディレクトリは最初に bind したプロセスが 0700 で作成します。

| 用途 | 既定パス（`$XDG_RUNTIME_DIR/std-t98/`） |
| --- | --- |
| frame | `std_t98_multi_frame.sock` |
| voice | `std_t98_multi_voice.sock` |
| status | `std_t98_multi_status.sock` |
| secret request | `std_t98_multi_secret_request.sock` |
| secret result | `std_t98_multi_secret_result.sock` |

置き場所は `STD_T98_RUNTIME_DIR` でまとめて、個別パスは `STD_T98_MULTI_FRAME_SOCKET` などで上書きできます（個別指定が優先）。backend と service を別々に起動する場合は、双方の `XDG_RUNTIME_DIR` が一致していることを確認してください（systemd unit や `sudo` 経由では未設定になり `/tmp` 側にずれます）。launcher 経由なら環境変数が継承されるため揃います。

## テスト

`env/` は launcher が service 用 Python として探すパスのため、テスト用の環境は `.venv/` に分けます。GNU Radio を見せるため `--system-site-packages` が必要です。

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

`tests/test_multi_audio_service.py` は音声系依存（sounddevice）を必要とします。RF 系のみの環境では `--ignore=tests/test_multi_audio_service.py` を付けてください。GNU Radio にも依存しないテストは `python3 -m unittest tests.test_backend_config` だけでも実行できます。音声系まで揃った環境なら `./env/bin/python -m pytest` でも同じものが回ります。

## ライセンス

GNU General Public License v3.0。詳細は [`LICENSE`](LICENSE) を参照してください。AMBE 復号は別リポジトリ [pyambelib](https://github.com/tallcat4/pyambelib)（mbelib-neo のラッパ）に依存します。
