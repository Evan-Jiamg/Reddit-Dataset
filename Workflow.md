# 專案工作流程：Reddit 民意量化 Pipeline

## 概覽

分析 Pushshift Reddit 留言資料集（2019 年 4 月，1.385 億筆），對特定議題的留言進行語義聚類，再用 fine-tune 過的 RoBERTa 模型對每則留言輸出 −1（反對）到 +1（支持）的連續立場分數。

---

## 專案結構

```
Twitter Dataset/
├── main.py                  # 步驟 0：下載並檢查資料集結構
├── eda.py                   # 步驟 1：全量串流 EDA（1.385 億筆）
├── visualize.py             # 步驟 2：200k 抽樣視覺化圖表
├── stage1_cluster.py        # 步驟 3：SBERT 語義聚類
├── stage2_train.py          # 步驟 4：RoBERTa 立場模型訓練（本機 CPU）
├── stage2_train_colab.ipynb # 步驟 4：RoBERTa 立場模型訓練（Colab GPU）
├── stage2_infer.py          # 步驟 5：對各 cluster 打立場分數
├── visualize_stance.py      # 步驟 6：立場分布視覺化
├── full_eda_report.txt      # 輸出：全量統計報告
├── Workflow.md              # 本文件
│
├── data/                    # 中間產物 / cache
│   ├── embeddings_politics.npy      # SBERT embedding cache（732 MB）
│   ├── ids_politics.json            # 留言 ID + body cache（119 MB）
│   ├── clusters_politics.json       # Stage 1 全量 cluster 定義（1,184 clusters）
│   ├── clusters_gun_abortion.json   # 篩選後：槍枝管制 + 墮胎（9 clusters）
│   └── clusters_politics_meta.json  # 聚類元資料
│
├── model/                   # 模型 checkpoint
│   ├── final_model.pt       # 最終訓練模型（476 MB）
│   └── cv_results.json      # 4-fold 交叉驗證結果
│
├── eda_charts/              # 全量 EDA 圖表（6 張）
│   ├── A_score_dist.png
│   ├── B_top_subreddits.png
│   ├── C_hourly.png
│   ├── D_body_length.png
│   ├── E_deletion_and_score.png
│   └── F_missing.png
│
├── stance_charts/           # 立場分析圖表（4 張）
│   ├── chart1_distribution.png
│   ├── chart2_per_cluster.png
│   ├── chart3_extremes.png
│   └── chart4_samples.png
│
└── stance_scores/           # 各 cluster 的 Parquet 輸出
    ├── cluster_132.parquet
    ├── cluster_131.parquet
    ├── cluster_203.parquet
    ├── cluster_456.parquet
    ├── cluster_705.parquet
    ├── cluster_708.parquet
    ├── cluster_725.parquet
    ├── cluster_750.parquet
    └── cluster_775.parquet
```

---

## 逐步說明

### 步驟 0 — 資料下載與結構檢查
**腳本**：`main.py`
**輸入**：Kaggle 資料集 `i221113hadiyatanveer/the-pushshift-reddit-dataset-submissions`
**輸出**：終端機印出資料夾結構與檔案資訊
**備註**：
- 主檔：`RC_2019-04.zst`（壓縮後 15.5 GB，Zstandard 格式）
- 每一行是一個 JSON 物件（一則 Reddit 留言）

---

### 步驟 1 — 全量串流 EDA
**腳本**：`eda.py`
**輸入**：完整 `RC_2019-04.zst`
**輸出**：`eda_report.txt`、`eda_charts/`（圖表 A–F）
**執行時間**：約 66 分鐘（35k 筆/秒，無 DataFrame，純串流）

**實作重點**：
- 以 8 MB chunk 串流讀取，逐行解析 JSON，不把整個檔案載入記憶體
- 使用 Welford 線上算法計算均值/變異數（數值穩定、單次遍歷）
- 以 Counter 追蹤各 subreddit 的刪除率與平均分數
- 獨立作者數量上限 200 萬，避免記憶體爆炸

**主要發現**：
- 總計 **138,473,643 筆**，0 筆解析錯誤
- 每筆 43 個欄位；關鍵欄位：`body`、`score`、`subreddit`、`author`、`created_utc`
- 分數：均值=7.62，標準差=107.47，最小值=−9090，最大值=66762（嚴重右偏）
- 整體刪除率：**11.56%**
- 尖峰時段：UTC 17–19 時（北美下午）
- `removal_reason`、`archived` 近 100% 為空 → 下游步驟可直接丟棄
- r/NBAPlayOffsLiveStream、r/AKHANvsCRAWFORDLive → 100% 被刪（直播洗版，可排除）

---

### 步驟 2 — EDA 視覺化
**腳本**：`visualize.py`
**輸入**：前 200,000 筆（快速抽樣）
**輸出**：`chart1_score.png` 到 `chart7_deleted.png`

**圖表列表**：
1. 分數分布 + CDF
2. 留言數 Top 25 Subreddit
3. 留言長度分布
4. 各欄位缺失率
5. 每小時留言量（UTC）
6. 分數 vs 留言長度散點圖 + 各 subreddit 分數中位數
7. 整體及各 subreddit 刪除率

---

### 步驟 3 — Stage 1：SBERT 語義聚類
**腳本**：`stage1_cluster.py`
**目標**：在指定 subreddit（如 r/politics）內，將留言依語義分群，讓同一 cluster 的留言討論同一個具體議題。

**輸入**：
- 從 `RC_2019-04.zst` 串流過濾特定 subreddit 的留言
- 過濾條件：非刪除、`body` 長度 > 20 字元

**模型**：`sentence-transformers/all-MiniLM-L6-v2`

**流程**：
1. 每則留言 body → 384 維向量（SBERT bi-encoder）
2. 用 **HDBSCAN** 聚類（`min_cluster_size` 可調整）
3. 每個 cluster 用 TF-IDF 提取關鍵字 → 自動生成 `topic_label` 與 `topic_description`

**粒度目標**：每個 cluster ≥ 50 則留言（`--min-cluster-size 50`，可手動調整）

**輸出格式**（`clusters_<subreddit>.json`）：
```json
{
  "cluster_id": 42,
  "topic_label": "trump immigration wall funding",
  "topic_description": "Discussion about: trump wall, immigration policy, border funding",
  "comment_ids": ["ejualnb", "ejualnd", "..."],
  "size": 312
}
```

**執行方式**：
```bash
python stage1_cluster.py --subreddit politics --min-cluster-size 50
```

---

### 步驟 4 — Stage 2：RoBERTa 立場模型訓練
**腳本**：`stage2_train.py`
**目標**：Fine-tune RoBERTa，輸出連續立場分數 ∈ [−1, +1]，代表反對 → 中立 → 支持。

#### 模型架構
```
輸入：[CLS] <topic_description> [SEP] <comment_body> [SEP]
              ↓
     RoBERTa-base（12 層，768 維）
              ↓
     Linear(768 → 1)
              ↓
     Tanh 激活
              ↓
輸出：純量 ∈ [−1, +1]
```

這是 **cross-encoder** 架構，topic 與 comment 跨所有 transformer 層互相 attend，能捕捉相對特定議題的細緻立場。

#### 訓練資料：IBM Argument Quality Ranking 30K
來源：HuggingFace `ibm-research/argument_quality_ranking_30k`（`datasets` 自動下載）

- **30,497 個論點**，橫跨 **71 個辯論議題**
- 關鍵欄位：
  - `argument`：論點文本
  - `topic`：議題描述
  - `stance_WA`：立場標籤（−1 反對 / +1 支持）
  - `stance_WA_conf`：標籤信心度 0–1（預設過濾 < 0.8）

直接使用 `stance_WA` 作為回歸目標，不需要偽標籤轉換。

**評估**：4-fold 跨議題驗證（GroupKFold by topic），確保模型能泛化到未見過的議題。

**執行方式**：
```bash
python stage2_train.py --epochs 3 --batch-size 16
# UKP_ASPECT.tsv 會自動從 GitHub 下載至 ukp_data/
```

**輸出**：
- `model/final_model.pt`（完整訓練模型）
- `model/cv_results.json`（各 fold 的 MSE/RMSE）

---

### 步驟 5 — Stage 2：對各 Cluster 進行立場推論
**腳本**：`stage2_infer.py`
**輸入**：
- Stage 1 輸出的 cluster JSON（`topic_description` + comment ID 列表）
- Stage 2 訓練的 RoBERTa checkpoint

**流程**：
1. 一次性串流掃描 `RC_2019-04.zst`，取出所有 cluster 需要的留言原文
2. 對每則留言構造輸入：`[CLS] {topic_description} [SEP] {comment_body} [SEP]`
3. 前向傳播 → 立場分數 ∈ [−1, +1]
4. 儲存：`(comment_id, stance, reddit_score, subreddit, created_utc)`

**輸出**：每個 cluster 一個 Parquet 檔案（`stance_scores/cluster_<id>.parquet`）

**執行方式**：
```bash
python stage2_infer.py \
    --clusters data/clusters_politics.json \
    --checkpoint model/final_model.pt
```

---

## 完整執行順序

```bash
# 步驟 0：確認資料
python main.py

# 步驟 1：全量 EDA（可選，約 66 分鐘）
python eda.py

# 步驟 2：抽樣圖表（快速）
python visualize.py

# 步驟 3：聚類（調整 --min-cluster-size 控制粒度）
python stage1_cluster.py --subreddit politics --min-cluster-size 50

# 步驟 4：訓練立場模型
python stage2_train.py --epochs 3 --batch-size 16

# 步驟 5：對每個 cluster 打分數
python stage2_infer.py \
    --clusters data/clusters_politics.json \
    --checkpoint model/final_model.pt
```

---

## 關鍵設計決策

| 決策 | 選擇 | 理由 |
|---|---|---|
| 聚類模型 | SBERT bi-encoder | 速度快，可處理百萬級留言 |
| 立場模型 | RoBERTa cross-encoder | 精度高，topic 與 comment 跨層互 attend |
| Stage 2 輸入格式 | `[topic][SEP][comment]` | 模型在評分時看到議題脈絡 |
| 輸出激活 | Tanh | 自然對應 [−1, +1] 範圍，可微分 |
| Cluster 大小門檻 | ≥ 50 則留言 | 統計上有意義；人工複查可行 |
| 刪除留言處理 | Stage 2 排除 | 無可用立場訊號 |
| 訓練資料 | UKP Argument Similarity（28 議題） | 連續標籤，支援跨議題驗證 |

---

## 已知限制

1. **諷刺/反語**：SBERT 與 RoBERTa 對 Reddit 風格的諷刺文本偵測能力有限，高爭議性 subreddit 可能出現偏高的中性分數。
2. **快照偏差**：Reddit 分數是爬取當下的數值，較早的留言累積分數通常更高，與留言品質不完全相關。
3. **僅支援英文**：模型以英文論點資料訓練，多語言 subreddit 中的非英文留言會產生不可靠的分數。
4. **Subreddit 選擇偏差**：每個 subreddit 有自己的意識形態傾向，分數反映的是**該社群**的民意，不代表一般大眾。
5. **議題描述品質**：Stage 2 分數對 `topic_description` 的措辭敏感，模糊或帶有立場的描述會影響所有下游分數。
