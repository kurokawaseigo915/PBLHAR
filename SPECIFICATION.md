# 姿勢キーポイント異常活動認識システム仕様書

## 1. 文書情報

| 項目 | 内容 |
|---|---|
| システム名 | Pose-based Unusual Activity Recognition |
| 対象実装 | `src/activity_model.py` |
| チューニング実装 | `src/tune_activity_model.py` |
| 作成日 | 2026-06-12 |
| 使用言語 | Python 3 |
| 主なライブラリ | NumPy、pandas、scikit-learn、joblib |

## 2. 目的

本システムは、匿名化された2次元姿勢キーポイント時系列だけを使用し、次の2つを実現する。

1. 各フレームの活動を8種類のいずれかに分類し、通常活動と異常活動を区別する。
2. Leave-One-Subject-Out（LOSO）評価により、学習に含まれない被験者への汎化性能を測定する。

## 3. 元Notebookとの関係

次のNotebookは提供時のまま保持し、変更していない。

- `SHLTutorial-main/01_data_loading.ipynb`
- `SHLTutorial-main/02_visualization.ipynb`
- `SHLTutorial-main/03_feature_extraction.ipynb`
- `SHLTutorial-main/04_model_training.ipynb`

`keypointlabel` データへの対応は、Notebookの直接改修ではなく、新規Python実装として作成した。

| Notebookの処理 | 新規実装での対応 |
|---|---|
| データ読込 | `load_pose()`、`load_subjects()` |
| 可視化 | 学習処理には含めない |
| 特徴抽出 | `normalize_pose()`、`window_features()` |
| モデル学習 | `fit_models()`、`train()` |
| 評価 | `evaluate()` |
| 未知データ予測 | `predict()` |

## 4. 使用データ

### 4.1 学習・LOSO評価データ

学習には、次のラベル付き姿勢キーポイントCSVだけを使用する。

```text
drive-download-20260612T052440Z-3-001/keypointlabel/
├── keypoints_with_labels_1.csv
├── keypoints_with_labels_2.csv
├── keypoints_with_labels_3.csv
└── keypoints_with_labels_5.csv
```

被験者IDは `1`、`2`、`3`、`5` の4名である。

### 4.2 推論データ

ラベルなし姿勢データを予測対象とする。

```text
drive-download-20260612T052440Z-3-001/keypoint/video_*.csv
```

### 4.3 使用しないデータ

次のデータは最終モデルの学習には使用しない。

- `timetable/csv/*.csv`
- `timetable/srt/*.srt`
- PDF資料
- Notebook内のサンプルデータや学習結果

## 5. 入力CSV仕様

### 5.1 共通列

入力CSVは、17個の関節について `x`、`y` 座標を持つ必要がある。

```text
nose
left_eye, right_eye
left_ear, right_ear
left_shoulder, right_shoulder
left_elbow, right_elbow
left_wrist, right_wrist
left_hip, right_hip
left_knee, right_knee
left_ankle, right_ankle
```

各関節は `<関節名>_x`、`<関節名>_y` の列名で表現する。座標列は合計34列である。

### 5.2 ラベル付きCSV

学習・評価用CSVには、共通列に加えて次の列が必要である。

| 列名 | 内容 |
|---|---|
| `frame_id` | フレーム番号 |
| `Action Label` | 活動ラベル |

### 5.3 ラベルなしCSV

推論用CSVでは `Action Label` は不要である。時刻識別には、次の優先順位で列を使用する。

1. `timestamp`
2. `Timestamp`
3. `frame_id`
4. いずれもない場合は30 fpsとして時刻を生成する

## 6. 活動クラス仕様

### 6.1 通常活動

- `Sitting quietly`
- `Using phone`
- `Walking`
- `Eating snacks`

### 6.2 異常活動

- `Head banging`
- `Throwing things`
- `Attacking`
- `Biting nails`

### 6.3 ラベル正規化

データ内の表記揺れを次のように統一する。

| 入力ラベル | 統一後 |
|---|---|
| `Biting` | `Biting nails` |
| `biting` | `Biting nails` |
| `biting nails` | `Biting nails` |
| `Throwing` | `Throwing things` |
| `throwing` | `Throwing things` |

8クラスに含まれないラベルおよび未ラベル区間は、学習と評価から除外する。

## 7. 前処理仕様

### 7.1 フレーム単位の姿勢正規化

各フレームについて、左右の腰の中点を姿勢の原点とする。

```text
腰中心 = (左腰座標 + 右腰座標) / 2
```

スケールには、胴体長と肩幅の大きい方を使用する。

```text
スケール = max(肩中心と腰中心の距離, 左右肩の距離)
正規化座標 = (元座標 - 腰中心) / スケール
```

これにより、画像上の絶対位置、身体サイズ、カメラとの距離の影響を軽減する。

### 7.2 時間窓

| 設定 | デフォルト値 |
|---|---:|
| 窓長 | 60フレーム（約2秒） |
| ストライド | 30フレーム（約1秒） |
| 想定フレームレート | 30 fps |

学習時は、同じラベルが連続する区間の内部だけで窓を生成する。異なる活動をまたぐ窓は作成しない。

入力が60フレーム未満の場合、推論時は末尾フレームを複製して60フレームまで補完する。

## 8. 特徴量仕様

各時間窓から次の情報を計算する。

1. 正規化した全17関節の座標
2. 上半身13関節間のユークリッド距離
3. 1フレーム差分による速度
4. 速度差分による加速度

座標と関節間距離を結合した時系列に対して、次の統計量を特徴量とする。

- 平均
- 標準偏差
- 最小値
- 最大値
- 窓の最終値と開始値の差
- 速度絶対値の平均
- 速度の標準偏差
- 速度絶対値の最大値
- 加速度絶対値の平均

### 8.1 被験者単位の特徴正規化

特徴抽出後、各被験者の全時間窓を対象に、特徴次元ごとのz-score正規化を行う。

```text
正規化特徴 = (特徴 - 被験者内平均) / 被験者内標準偏差
```

この処理は活動ラベルを使用しない。未知被験者の推論時も、その被験者の入力姿勢特徴の平均と標準偏差だけを使用する。

## 9. モデル仕様

### 9.1 モデル構成

scikit-learnの `ExtraTreesClassifier` を2つ使用する。

| モデル | 目的 | `n_estimators` | `min_samples_leaf` | `max_features` | `class_weight` |
|---|---|---:|---:|---|---|
| 活動分類モデル | 8クラス分類 | 300 | 4 | `sqrt` | なし |
| 異常判定モデル | Normal / Unusual分類 | 300 | 2 | `sqrt` | なし |

共通の乱数シードは `42`、並列実行設定は `n_jobs=-1` とする。

### 9.2 階層予測

1. 活動分類モデルが8クラスの活動を予測する。
2. 異常判定モデルが `Normal` または `Unusual` を予測する。
3. 2つの予測グループが一致しない場合、異常判定モデルの結果を優先する。
4. 活動分類モデルの確率から、指定されたグループ内で最も確率の高い活動を最終ラベルとする。

例として、活動分類が `Walking`、異常判定が `Unusual` の場合、異常4クラスの中で最も確率が高い活動へ修正する。

## 10. LOSO評価仕様

4名の被験者を順番に1名ずつ評価対象として除外する。

| Fold | 学習対象 | 評価対象 |
|---|---|---|
| 1 | 被験者2、3、5 | 被験者1 |
| 2 | 被験者1、3、5 | 被験者2 |
| 3 | 被験者1、2、5 | 被験者3 |
| 4 | 被験者1、2、3 | 被験者5 |

評価対象被験者の時間窓やラベルは、当該Foldのモデル学習には使用しない。

### 10.1 評価指標

- 8クラス Accuracy
- 8クラス Macro-F1
- 異常活動 Precision
- 異常活動 Recall
- 異常活動 F1
- Normal / Unusual混同行列
- 8クラス分類レポート
- 8クラス混同行列

### 10.2 現在の評価結果

| 評価対象 | Accuracy | Macro-F1 | 異常Precision | 異常Recall | 異常F1 |
|---|---:|---:|---:|---:|---:|
| 被験者1 | 0.6660 | 0.6761 | 0.9803 | 0.7393 | 0.8429 |
| 被験者2 | 0.7157 | 0.6423 | 0.9209 | 0.7098 | 0.8017 |
| 被験者3 | 0.5294 | 0.4591 | 0.6830 | 0.4918 | 0.5718 |
| 被験者5 | 0.6552 | 0.7153 | 0.9978 | 0.7722 | 0.8706 |
| 全体 | **0.6295** | **0.6279** | **0.8825** | **0.6634** | **0.7574** |

## 11. 出力仕様

### 11.1 予測CSV

予測結果は次の3列で出力する。

```text
participant_id,timestamp,predicted_label
```

| 列名 | 内容 |
|---|---|
| `participant_id` | コマンド引数または入力ファイル名から取得した被験者ID |
| `timestamp` | 入力の時刻、フレーム番号、または30 fpsで生成した時刻 |
| `predicted_label` | 8活動のいずれか |

時間窓単位の予測は、隣接する窓中心の中点を境界として全フレームへ展開する。

### 11.2 学習済みモデル

joblib形式の辞書として保存する。

| キー | 内容 |
|---|---|
| `activity_model` | 8クラスExtra Treesモデル |
| `abnormal_model` | Normal / Unusual Extra Treesモデル |
| `config` | 窓長、ストライド、木の数などの設定 |
| `feature_normalization` | `subject_zscore` |
| `model_type` | `pose_subject_normalized_v2` |

### 11.3 LOSO評価JSON

次の情報を保存する。

- 実行設定
- Fold別評価結果
- 全体Accuracy
- 全体Macro-F1
- 全体異常活動指標
- 分類レポート
- 混同行列

## 12. 実行仕様

### 12.1 環境構築

```bash
cd /Users/seigokuro/Downloads/PBL
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 12.2 LOSO評価

```bash
.venv/bin/python src/activity_model.py evaluate \
  --data-dir drive-download-20260612T052440Z-3-001/keypointlabel \
  --output outputs/loso_metrics.json
```

### 12.3 最終モデル学習

```bash
.venv/bin/python src/activity_model.py train \
  --data-dir drive-download-20260612T052440Z-3-001/keypointlabel \
  --output outputs/activity_model.joblib
```

### 12.4 未知被験者の予測

```bash
.venv/bin/python src/activity_model.py predict \
  --model outputs/activity_model.joblib \
  --input path/to/test_keypoints.csv \
  --participant test_subject_id \
  --output outputs/A.csv
```

### 12.5 設定変更

評価と学習では次の引数を指定できる。

| 引数 | デフォルト | 内容 |
|---|---:|---|
| `--window` | 60 | 窓長 |
| `--stride` | 30 | ストライド |
| `--trees` | 300 | 決定木数 |
| `--normalization` | `subject_zscore` | 特徴正規化方式 |
| `--activity-leaf` | 4 | 活動分類モデルの最小葉サイズ |
| `--abnormal-leaf` | 2 | 異常判定モデルの最小葉サイズ |

## 13. ファイル構成

| ファイル | 役割 |
|---|---|
| `src/activity_model.py` | 読込、特徴抽出、評価、学習、予測の本体 |
| `src/tune_activity_model.py` | LOSOによる設定比較 |
| `requirements.txt` | Python依存ライブラリ |
| `outputs/activity_model.joblib` | 正式な改善版モデル |
| `outputs/loso_metrics.json` | 正式なLOSO評価結果 |
| `outputs/A.csv` | 予測結果 |
| `README.md` | 簡易実行手順 |
| `REPORT.md` | 手法、評価結果、考察 |
| `SPECIFICATION.md` | 本仕様書 |

## 14. エラー条件

次の場合は処理を中断する。

- 必須の34座標列が不足している
- 学習用CSVに `Action Label` が存在しない
- 学習ディレクトリ内の被験者数が2名未満
- 未対応の特徴正規化方式が指定された
- モデルファイルまたは入力CSVが存在しない

## 15. 制約と注意事項

- 現在の評価は時間窓単位であり、イベント単位の検出評価ではない。
- 学習対象は4名であり、被験者数が少ないため、新しい撮影環境で同等の性能を保証するものではない。
- 被験者3は他被験者より性能が低く、カメラ位置や動作表現の差が残っている可能性がある。
- 推論時のz-scoreは入力ファイル全体から計算するため、短い入力や単一活動だけの入力では正規化が不安定になる場合がある。
- 現在の姿勢CSVにはキーポイント信頼度列がないため、関節検出の信頼度を特徴量として利用していない。
- `outputs/A.csv` は現在存在するpose-onlyファイルに対する出力である。正式な未知被験者データが提供された場合は、同じ `predict` コマンドで改めて生成する。

## 16. 改善前モデルとの差分

| 項目 | 改善前 | 現在 |
|---|---|---|
| 被験者単位特徴正規化 | なし | z-score |
| 活動分類の最小葉サイズ | 2 | 4 |
| 異常判定の最小葉サイズ | 2 | 2 |
| クラス重み | `balanced` | なし |
| Accuracy | 0.5500 | 0.6295 |
| Macro-F1 | 0.5138 | 0.6279 |
| 異常F1 | 0.6279 | 0.7574 |

元のCSV、ラベル、Notebook、タイムテーブル、参考資料は変更していない。変更・追加対象は、Python実装、文書、学習済みモデル、評価結果、予測結果だけである。
