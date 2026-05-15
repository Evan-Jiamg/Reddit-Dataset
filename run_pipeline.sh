#!/usr/bin/env bash
# Full pipeline: Stage 1 → auto-identify Gun/Abortion clusters → Stage 2 → Visualize
set -e

PYTHON="/mnt/NewSSD/CS_project/reddit_env/bin/python3"   # GPU cuml env
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ZST="/mnt/NewSSD/CS_project/kagglehub/datasets/i221113hadiyatanveer/the-pushshift-reddit-dataset-submissions/versions/1/RC_2019-04.zst"
OUT_DATA="/mnt/NewSSD/CS_project/Reddit-Dataset/data"
OUT_SCORES="/mnt/NewSSD/CS_project/Reddit-Dataset/stance_scores"
CLUSTERS_ALL="$OUT_DATA/clusters_politics.json"
CLUSTERS_SUB="$OUT_DATA/clusters_gun_abortion.json"
CKPT="$SCRIPT_DIR/model/final_model.pt"

export REDDIT_DATA_PATH="$DATA_ZST"

echo "================================================================"
echo " Reddit Stance Pipeline — $(date)"
echo " Python: $PYTHON"
echo "================================================================"

# ── Stage 1 ──────────────────────────────────────────────────────────
echo ""
echo ">>> Stage 1: SBERT + HDBSCAN clustering (GPU UMAP via cuml)"
"$PYTHON" "$SCRIPT_DIR/stage1_cluster.py" \
  --subreddit politics \
  --max-per-author 3 \
  --max-comments 500000 \
  --output-dir "$OUT_DATA" \
  --umap-dims 50

echo ""
echo ">>> Stage 1 complete. Identifying Gun Control + Abortion clusters..."

# ── Auto-identify Gun / Abortion clusters ────────────────────────────
"$PYTHON" - <<'PYEOF'
import json, re, os

GUN_KEYWORDS  = {"gun", "guns", "firearm", "firearms", "nra", "rifle",
                 "pistol", "weapon", "weapons", "amendment", "2nd", "second"}
ABRT_KEYWORDS = {"abortion", "abortions", "pro-life", "pro-choice", "roe",
                 "fetus", "reproductive", "planned parenthood"}

clusters_path = os.environ.get("CLUSTERS_ALL",
    "/mnt/NewSSD/CS_project/Reddit-Dataset/data/clusters_politics.json")
out_path = os.environ.get("CLUSTERS_SUB",
    "/mnt/NewSSD/CS_project/Reddit-Dataset/data/clusters_gun_abortion.json")

with open(clusters_path, encoding="utf-8") as f:
    clusters = json.load(f)

gun_hits, abrt_hits = [], []
for c in clusters:
    label = c["topic_label"].lower()
    words = set(re.findall(r"\w+", label))
    if words & GUN_KEYWORDS:
        gun_hits.append(c)
    elif words & ABRT_KEYWORDS:
        abrt_hits.append(c)

print(f"  Found {len(gun_hits)} Gun Control clusters:")
for c in gun_hits:
    print(f"    id={c['cluster_id']}  size={c['size']}  {c['topic_label'][:60]}")
print(f"  Found {len(abrt_hits)} Abortion clusters:")
for c in abrt_hits:
    print(f"    id={c['cluster_id']}  size={c['size']}  {c['topic_label'][:60]}")

subset = gun_hits + abrt_hits
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)
print(f"\n  Saved {len(subset)} clusters → {out_path}")
PYEOF

export CLUSTERS_ALL CLUSTERS_SUB

# ── Stage 2 Training (if no model checkpoint exists) ─────────────────
if [ ! -f "$CKPT" ]; then
    echo ""
    echo ">>> Stage 2 Training: fine-tuning RoBERTa on IBM ArgQ-30k (RTX 4090)"
    "$PYTHON" "$SCRIPT_DIR/stage2_train.py" \
      --epochs 3 \
      --lr 2e-5 \
      --batch-size 64 \
      --max-length 256 \
      --folds 4 \
      --min-conf 0.8
else
    echo ""
    echo ">>> Stage 2 Training: skipped (checkpoint exists: $CKPT)"
fi

# ── Stage 2 Inference ─────────────────────────────────────────────────
echo ""
echo ">>> Stage 2: Stance inference on Gun/Abortion clusters"
mkdir -p "$OUT_SCORES"
"$PYTHON" "$SCRIPT_DIR/stage2_infer.py" \
  --clusters "$CLUSTERS_SUB" \
  --checkpoint "$CKPT" \
  --output-dir "$OUT_SCORES" \
  --batch-size 64 \
  --max-length 256

# ── Auto-update visualize_stance.py cluster IDs ──────────────────────
echo ""
echo ">>> Updating cluster IDs in visualize_stance.py..."
"$PYTHON" - <<'PYEOF'
import json, os, re

clusters_path = "/mnt/NewSSD/CS_project/Reddit-Dataset/data/clusters_gun_abortion.json"
vis_path = "/home/evan-jiamg/Reddit-Dataset/Reddit-Dataset/visualize_stance.py"

GUN_KEYWORDS  = {"gun", "guns", "firearm", "firearms", "nra", "rifle",
                 "pistol", "weapon", "weapons", "amendment", "2nd", "second"}
ABRT_KEYWORDS = {"abortion", "abortions", "pro-life", "pro-choice", "roe",
                 "fetus", "reproductive", "planned"}

with open(clusters_path, encoding="utf-8") as f:
    clusters = json.load(f)

gun_ids, abrt_ids = [], []
cluster_meta = {}
for c in clusters:
    label = c["topic_label"].lower()
    words = set(re.findall(r"\w+", label))
    cid = c["cluster_id"]
    sz  = c["size"]
    if words & GUN_KEYWORDS:
        gun_ids.append(cid)
        cluster_meta[cid] = ("Gun Control", sz)
    elif words & ABRT_KEYWORDS:
        abrt_ids.append(cid)
        cluster_meta[cid] = ("Abortion", sz)

gun_set  = "{" + ", ".join(str(i) for i in sorted(gun_ids))  + "}"
abrt_set = "{" + ", ".join(str(i) for i in sorted(abrt_ids)) + "}"

meta_lines = []
for cid, (topic, sz) in sorted(cluster_meta.items()):
    meta_lines.append(f"    {cid}: (\"{topic}\", {sz}),")
meta_block = "CLUSTER_META = {\n" + "\n".join(meta_lines) + "\n}"

with open(vis_path, encoding="utf-8") as f:
    src = f.read()

src = re.sub(r"GUN_IDS\s*=\s*\{[^}]*\}",      f"GUN_IDS      = {gun_set}",  src)
src = re.sub(r"ABORTION_IDS\s*=\s*\{[^}]*\}",  f"ABORTION_IDS = {abrt_set}", src)
src = re.sub(r"CLUSTER_META\s*=\s*\{[^}]*\}", meta_block, src, flags=re.DOTALL)
src = src.replace(
    'SCORES_DIR = os.path.join(SCRIPT_DIR, "stance_scores")',
    'SCORES_DIR = "/mnt/NewSSD/CS_project/Reddit-Dataset/stance_scores"'
)
src = src.replace(
    'IDS_CACHE  = os.path.join(SCRIPT_DIR, "data", "ids_politics.json")',
    'IDS_CACHE  = "/mnt/NewSSD/CS_project/Reddit-Dataset/data/ids_politics.json"'
)

with open(vis_path, "w", encoding="utf-8") as f:
    f.write(src)

print(f"  GUN_IDS      = {gun_set}")
print(f"  ABORTION_IDS = {abrt_set}")
print(f"  CLUSTER_META updated ({len(cluster_meta)} entries)")
print(f"  Patched: {vis_path}")
PYEOF

# ── Visualize ────────────────────────────────────────────────────────
echo ""
echo ">>> Generating charts..."
"$PYTHON" "$SCRIPT_DIR/visualize_stance.py"

echo ""
echo "================================================================"
echo " Pipeline complete! Charts saved to:"
echo "   $SCRIPT_DIR/stance_charts/"
echo " $(date)"
echo "================================================================"
