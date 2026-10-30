# AI×ロボティクスによる創造的タスクの実現: 洗濯物を洗濯機へ運ぶロボット

## 概要

Genesis 1.4.4 と IPC（libuipc）で、小さな 1K の住居（寸法は `tools/params.py` の大まかな推定値）をシミュレーションで再現した。
ホロノミック台車・昇降軸・PiPER アーム・かごを持つロボットが、洋室から廊下を通って縦型洗濯機まで走り、かごの洗濯物を槽へ投入する。
AI 部分は自前ホストの VLM（ollama 上の `qwen3-vl:8b-instruct`）で、かごのカメラ画像から洗濯物を見つける。

パイプライン:

1. かごに洗濯物を落とし、静止するまで待つ。ロボットは洋室から廊下を通って洗濯機の前まで走る。
2. かご上のカメラ画像を VLM に渡し、洗濯物ごとの bbox を得る。VLM が見落とした洗濯物色の領域には再質問（recheck）する。
3. 検出内の点群（深度）から把持点とグリッパのヨーを決め、一番上にあるものから順に選ぶ。
4. 計画した腕の動きでつまみ（摩擦ピンチ）、持ち上げ、洗濯機の上まで運んで放す。
5. 槽を確認し、かごが空になるか、残りが全て諦め対象になるまで 2 から繰り返す。

シナリオは 2 つ: タオル 1 枚（`scripts/integrated_demo.py`）と、タオル・T シャツ・靴下の 3 点（`scripts/multi_integrated_demo.py`）。

## 動画

YouTube: TODO

## 構成

```text
assets/            URDF とテクスチャ（room/ washer/ robot/ textures/）。URDF は tools/ の生成物
  robot/piper/     PiPER 公式モデル（MIT）
tools/             アセット生成（params.py が寸法の定義、gen_*.py、gen_all.py、check_assets.py）
scripts/           シミュレーション・VLM・計画・動画のスクリプト
  integrated_demo.py        タオル 1 枚の統合テイク
  multi_integrated_demo.py  3 点の統合テイク
  vlm_grasp.py, vlm_multi.py  VLM による検出と把持点
  arm_planner.py            アームの経路計画
  make_final_video.py       最終動画の合成
  render_robot.py           ロボット単体の描画（名前付き。動画の「ロボット構成と参考価格」のカード用）
  remote/                   リモート GPU 機での実行補助（setup.sh, sync_to.sh, sync_from.sh, collect_and_stop.sh ほか）
docs/final_video/  最終動画に載せる統計（stats_ja.json, stats_multi_ja.json）と部品の参考価格（parts_ja.json）
outputs/           実行結果（gitignore 済み。リポジトリには含まれない）
```

## ロボットの部品と参考価格

シミュレーションのロボット（台車 0.45 m 角、昇降 0〜0.5 m、PiPER、籠、籠を真上から撮る RGB-D カメラ）を実際に組むとしたときの、寸法と仕様が近い市販品の例を挙げる。価格は 2026 年 10 月 30 日に各販売ページで見た表示価格で、動画の「ロボット構成と参考価格」のカードと同じ値を使っている（`docs/final_video/parts_ja.json`）。実機は作っていない。

| 部品 | 製品の例 | 仕様（モデルとの対応） | 価格 | 出典 |
| ---- | -------- | ---------------------- | ---- | ---- |
| 台車 | Wheeltec R5 Series Mecanum Wheel Chassis（13-wire Hall encoder） | メカナム 4 輪、約 0.41 m 角、積載 15 kg | $454.95 | https://openelab.com/products/wheeltec-r5-series-robot-mecanum |
| 昇降軸 | Progressive Automations LG-02（LG-02-3RM-25-180-HS-24V） | 電動コラム、ストローク 648 mm、180 lb | $115.99 | https://www.progressiveautomations.com/products/lg-02 |
| アーム | AgileX PiPER | 6 自由度、可搬 1.5 kg、リーチ 626 mm | $1,999.00 | https://global.agilex.ai/products/piper |
| カメラ | RealSense Depth Camera D435i | 深度 640×480、深度の画角 87°×58° | $399.00 | https://store.realsenseai.com/buy-realsense-depth-camera-d435i.html |
| 合計 | | | 約 $2,969（約 47 万円） | 1 ドル = 157.54 円（三菱UFJ銀行の 2026-10-30 の公表相場 TTS 158.54 / TTB 156.54 の仲値）: https://www.murc-kawasesouba.jp/fx/past/index.php?id=261030 |

- 含まないもの: PiPER のグリッパ（別売）、台車の駆動回路と電池、昇降軸の制御箱（同社の FLTCON-2-24VDC は $146.99）、VLM を動かす計算機。
- 税・送料の扱いは店ごとに異なる（台車のページは関税込みと表示、カメラの店は 2026-02-03 から関税分の追加料金があり、額は表示されていない）。
- PiPER の価格は、ページに表示がなく、店の商品データの値（$1,999.00）である。国内では 2024-11-20 の発売時に本体 398,000 円（税別、発売記念価格）と発表されている: https://www.dreamnews.jp/press/0000308998/
- 台車の積載 15 kg は、アーム 4.2 kg・昇降コラム 6.8 kg・籠・電池を載せるとほぼ上限になる。

## 環境

- GPU: ローカルは RTX 5070 Ti 16 GB、リモートは RTX 5060 Ti 16 GB。どちらも Blackwell のため CUDA 12.8 以降（torch cu128）が必要。
- Python 3.12 と [uv](https://docs.astral.sh/uv/)。環境は 2 つに分けている。
  - `.venv`（Genesis のみ。アセット確認・静止画・CPU 確認用）: `uv sync`（`pyproject.toml` と `uv.lock`。torch は cu128 インデックス）。
  - `.venv-ipc`（IPC の `pyuipc` を含む。統合デモはこちら）: `requirements-ipc.txt` の冒頭の手順どおり、torch を単独で先に入れてから残りを入れる。

    ```sh
    uv venv .venv-ipc --python 3.12
    uv pip install --python .venv-ipc/bin/python --no-deps \
        --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
    uv pip install --python .venv-ipc/bin/python -r requirements-ipc.txt
    ```
- VLM: [ollama](https://ollama.com/) を起動し、`ollama pull qwen3-vl:8b-instruct`（約 6 GB、VRAM は約 7.4 GB）。
- 新しい GPU 機への一括セットアップは `bash scripts/remote/setup.sh`（apt、uv、両 venv、ollama、スモークテスト。`--skip-system --skip-ollama --skip-ipc` あり）。
  リモート実行の流れは `scripts/remote/README.md` を参照（`scripts/remote/sync_to.sh`, `scripts/remote/sync_from.sh`, `scripts/remote/collect_and_stop.sh`）。
- GPU は 1 枚を Genesis と ollama で共有するので、重い実行は `flock /tmp/house_robot-gpu.lock` で直列化する。

動作確認: `uv run python scripts/check_env.py`（初回はカーネルコンパイルで約 80 秒）。

## 実行方法

### アセットの再生成と検査（CPU、数秒）

```sh
python3 tools/gen_all.py
python3 tools/check_assets.py
```

### 共通フラグ

以降のコマンドで使う本番設定（`BASE`）:

```sh
BASE="--pre-open 0.015 --no-scoop --depth 0.03 --min-bite 0.015 --stall-check --eps-v 1e-3 \
--grip-kp 8000 --grip-fmax 40 --mu 1.2 --miss-check --newton-tol 1e-3 --attempts 5 \
--skip-carry-on-fail --carry-mode gentle --carry-s 9 --n-wp 6 --cs-trans 30 --cs-rot 30 \
--planner --grip-const --floor-clear 0.015 --ff grav+vel --semi-implicit off"
```

### タオル 1 枚

`scripts/integrated_demo.py` は `scripts/cloth_ipc_washer_test.py` のフラグを引き継ぐので、`BASE` に `--vlm ollama` を付ける。

```sh
flock /tmp/house_robot-gpu.lock .venv-ipc/bin/python scripts/integrated_demo.py --seed 0 $BASE --vlm ollama \
  --out outputs/integrated_demo.mp4 --json outputs/integrated_demo.json
```

### 3 点（タオル・T シャツ・靴下）

```sh
flock /tmp/house_robot-gpu.lock .venv-ipc/bin/python scripts/multi_integrated_demo.py --seed 0 --max-cycles 5 \
  --settle 3 --settle-max 6 $BASE --vlm ollama --out outputs/multi/r3_s0.mp4 --json outputs/multi/r3_s0.json
```

所要時間の目安は 1 テイクあたり約 1〜2 時間（RTX 5070 Ti・RTX 5060 Ti の 12 本で 62〜115 分。シミュレーション時間の約 46〜79 倍）。

### 最終動画の合成

```sh
uv run --no-sync python scripts/make_final_video.py --take outputs/multi/r3_s0.mp4 --json outputs/multi/r3_s0.json \
  --stats docs/final_video/stats_multi_ja.json --lang ja --out outputs/final/final_ja_multi_s0.mp4
```

動画はシミュレーション時間の等倍で、VLM の推論中は止めて、検出結果のカードを静止表示している。

## 結果

3 点の版（走行・動画あり、同じコードと同じ設定）の 12 本。投入数はシミュレーションの判定。

| take | GPU | 投入数 | 把持の試行 | 備考 |
| ---- | --- | ------ | ---------- | ---- |
| R3 s0 | RTX 5070 Ti | 3/3 | 2 | 1 回目に靴下とTシャツを同時に持ち上げて投入 |
| R3 s1 | RTX 5060 Ti | 3/3 | 3 | 1 回目に VLM が見落としたタオルを、観測し直して後で投入 |
| R3 s2 | RTX 5070 Ti | 2/3 | 2 | 靴下が運搬中に床へ落ちた |
| R4 s3 | RTX 5060 Ti | 3/3 | 1 | 1 回の把持で 3 着同時 |
| R4 s4 | RTX 5060 Ti | 3/3 | 4 | 1 着ずつ（空振り 1 回） |
| R4 s5 | RTX 5060 Ti | 3/3 | 2 | 2 回目にタオルと靴下を同時に投入 |
| R4 s6 | RTX 5070 Ti | 3/3 | 3 | 一緒に持ち上がったタオルが籠に戻り、次の回に投入 |
| R4 s7 | RTX 5070 Ti | 3/3 | 2 | 指に残ったタオルを、手首を振って槽へ落とした |
| R4 s8 | RTX 5070 Ti | 3/3 | 2 | 2 回目にタオルとTシャツを同時に投入 |
| R5 s9 | RTX 5060 Ti | 2/3 | 2 | 同時に持ち上げた靴下が手に残り、台車の上へ落ちた |
| R5 s10 | RTX 5060 Ti | 2/3 | 3 | 同時に持ち上げた靴下が運搬中に台車の上へ落ちた |
| R5 s11 | RTX 5060 Ti | 2/3 | 3 | 同時に持ち上げたTシャツが運搬中に床へ落ちた |
| 合計 | – | 32/36 | 29 | 3 着とも入ったのは 12 本中 8 本 |

籠の外に落ちた物は拾わない（床からの拾い上げは実装していない）。籠の中に戻った物は、毎回観測し直すので次の回に拾える。

タオル 1 枚の版（統合デモ）は、同じ設定の 6 本中 5 本で成功した（失敗の 1 本は、放した後もタオルが指に残った）。

## 正直に書いておくこと

- 3 点の版（`scripts/multi_integrated_demo.py`）では、掴む物・把持点・向きを、カメラ画像と深度の点群だけから決めている。シミュレータの真値は使っていない。
- タオル 1 枚の版（`scripts/integrated_demo.py`）では、VLM が選んだ点を探索の中心にし、その周りにある布の頂点（シミュレータの値）から把持点と向きを決めている。持ち上げの成否の判定にも布の頂点を使っている。
- 3 点の版では、つまみの失敗（空振り）・保持が弱い・放した後に指に残っている、の判定に、シミュレータの接触・力の値（全洗濯物の合計）を「模擬した触覚・力覚」として使っている。物体ごとの識別はしていない。
- 3 点の版では、シミュレータの真値を採点（槽に入った割合など）と、動画の「シミュレーション判定」と表示した字幕にだけ使っている。
- VLM が返す洗濯物の種類のラベルは当てにならない。つかむ順番は種類ではなく高さ（一番上から）で決めている。
- 住居は 1K の間取りを大まかな推定寸法だけでモデル化したもの。写真や図面はリポジトリに含めず、テクスチャは手続き生成である。

## ライセンス・素材

- `assets/robot/piper/`: PiPER の公式モデル（MIT、`assets/robot/piper/LICENSE`）。変更は `package://` パスの相対化のみ。
- その他のコード・アセット生成物・文書: MIT（リポジトリ直下の `LICENSE`）。
