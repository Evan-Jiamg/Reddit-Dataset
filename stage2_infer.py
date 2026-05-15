"""
Stage 2 Inference: Load clusters from Stage 1 + fine-tuned RoBERTa →
score each comment → output per-cluster Parquet files
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zstandard as zstd
from transformers import RobertaTokenizer, RobertaModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.environ.get(
    "REDDIT_DATA_PATH",
    os.path.join(
        os.path.expanduser("~"),
        ".cache", "kagglehub", "datasets",
        "i221113hadiyatanveer",
        "the-pushshift-reddit-dataset-submissions",
        "versions", "1", "RC_2019-04.zst"
    )
)


class StanceModel(nn.Module):
    def __init__(self, roberta_name: str = "roberta-base"):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_name)
        self.regressor = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return torch.tanh(self.regressor(cls)).squeeze(-1)


def load_model(ckpt_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device)
    roberta_name = ckpt.get("roberta_name", "roberta-base")
    model = StanceModel(roberta_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained(roberta_name)
    print(f"Loaded model from {ckpt_path}  (roberta={roberta_name})")
    return model, tokenizer


def stream_comment_index(data_path: str, target_ids: set[str]) -> dict[str, dict]:
    """Scan zst file and return metadata for comments matching target_ids."""
    found = {}
    with open(data_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while len(found) < len(target_ids):
                chunk = reader.read(8 * 1024 * 1024)
                if not chunk:
                    break
                buf += chunk
                lines = buf.split(b"\n")
                buf = lines[-1]
                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cid = rec.get("id") or rec.get("name", "")
                    if cid in target_ids:
                        found[cid] = {
                            "body": rec.get("body", ""),
                            "score": rec.get("score", 1),
                            "subreddit": rec.get("subreddit", ""),
                            "created_utc": rec.get("created_utc", 0),
                        }
    return found


def score_batch(texts: list[str], topic_desc: str, model, tokenizer,
                device: torch.device, batch_size: int, max_length: int) -> list[float]:
    scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(
            [topic_desc] * len(batch_texts),
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            preds = model(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            ).cpu().numpy()
        scores.extend(preds.tolist())
    return scores


def run(clusters_path: str, ckpt_path: str, output_dir: str,
        batch_size: int, max_length: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_model(ckpt_path, device)

    with open(clusters_path, encoding="utf-8") as f:
        clusters = json.load(f)
    print(f"Loaded {len(clusters)} clusters from {clusters_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Gather all needed comment IDs in one streaming pass
    all_ids = set()
    for c in clusters:
        all_ids.update(c["comment_ids"])
    print(f"Fetching {len(all_ids):,} comments from dataset...")
    comment_index = stream_comment_index(DATA_PATH, all_ids)
    print(f"  Found {len(comment_index):,} / {len(all_ids):,} comments")

    for c in clusters:
        cid_val = c["cluster_id"]
        topic_desc = c["topic_description"]
        comment_ids = c["comment_ids"]

        rows = []
        texts = []
        meta = []
        for cid in comment_ids:
            info = comment_index.get(cid)
            if info is None:
                continue
            body = info["body"]
            if body in ("[deleted]", "[removed]", "") or len(body) < 10:
                continue
            texts.append(body)
            meta.append({
                "comment_id": cid,
                "reddit_score": info["score"],
                "subreddit": info["subreddit"],
                "created_utc": info["created_utc"],
            })

        if not texts:
            continue

        stance_scores = score_batch(texts, topic_desc, model, tokenizer,
                                    device, batch_size, max_length)

        for m, stance, text in zip(meta, stance_scores, texts):
            rows.append({
                "comment_id": m["comment_id"],
                "stance": round(float(stance), 4),
                "reddit_score": m["reddit_score"],
                "subreddit": m["subreddit"],
                "created_utc": m["created_utc"],
                "cluster_id": cid_val,
                "topic_label": c["topic_label"],
            })

        df = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, f"cluster_{cid_val}.parquet")
        df.to_parquet(out_path, index=False)
        mean_stance = df["stance"].mean()
        print(f"  Cluster {cid_val:>4} ({len(df):>5} comments)  "
              f"mean_stance={mean_stance:+.3f}  → {os.path.basename(out_path)}")

    print(f"\nInference complete. Parquet files saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: stance inference on clusters")
    parser.add_argument("--clusters", required=True,
                        help="Path to clusters_<subreddit>.json from Stage 1")
    parser.add_argument("--checkpoint", default=os.path.join(SCRIPT_DIR,
                        "model", "final_model.pt"),
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "stance_scores"),
                        help="Directory for output Parquet files")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    run(
        clusters_path=args.clusters,
        ckpt_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
