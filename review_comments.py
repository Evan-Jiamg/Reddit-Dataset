"""
review_comments.py
──────────────────
檢視 stance_scores/ 的留言，並可刪除指定留言。

用法範例：
  # 列出槍枝議題分數最高的 20 則（最支持管制）
  python review_comments.py --topic gun --order desc --limit 20

  # 列出墮胎議題分數最低的 20 則（最反對合法化）
  python review_comments.py --topic abortion --order asc --limit 20

  # 只看特定 stance 範圍
  python review_comments.py --topic gun --min-stance 0.8 --max-stance 1.0

  # 列出特定 cluster
  python review_comments.py --cluster 775 --order asc --limit 10

  # 刪除指定 comment_id（可多個，空格分隔）
  python review_comments.py --delete ejuar4k ejuasfd ejub1rj

  # 顯示所有 cluster 的統計摘要
  python review_comments.py --summary
"""

import os
import json
import argparse
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SCORES_DIR  = os.path.join(SCRIPT_DIR, "stance_scores")
IDS_CACHE   = os.path.join(SCRIPT_DIR, "data", "ids_politics.json")

GUN_IDS      = {775, 708, 750, 705, 456, 725}
ABORTION_IDS = {132, 131, 203}

TOPIC_MAP = {c: "Gun Control" for c in GUN_IDS}
TOPIC_MAP.update({c: "Abortion" for c in ABORTION_IDS})


def load_all(load_bodies: bool = True) -> pd.DataFrame:
    frames = []
    for fname in sorted(os.listdir(SCORES_DIR)):
        if not fname.endswith(".parquet"):
            continue
        cid = int(fname.replace("cluster_", "").replace(".parquet", ""))
        df = pd.read_parquet(os.path.join(SCORES_DIR, fname))
        df["topic"] = TOPIC_MAP.get(cid, "Unknown")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    if load_bodies and os.path.exists(IDS_CACHE):
        with open(IDS_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
        body_map = dict(zip(cache["ids"], cache["bodies"]))
        data["body"] = data["comment_id"].map(body_map)
    else:
        data["body"] = "—"

    return data


def print_comments(df: pd.DataFrame, limit: int, body_width: int = 200):
    print(f"\n{'#':>4}  {'Score':>7}  {'Cluster':>9}  {'Topic':<14}  Comment")
    print("─" * 90)
    for i, (_, r) in enumerate(df.head(limit).iterrows(), 1):
        body = str(r.get("body", "—"))
        snippet = body[:body_width].replace("\n", " ")
        if len(body) > body_width:
            snippet += "…"
        print(f"{i:>4}  {r.stance:>+7.3f}  #{r.cluster_id:<8}  {r.topic:<14}  {snippet}")
    print(f"\n  ({min(limit, len(df))} of {len(df):,} shown)")


def show_summary(data: pd.DataFrame):
    print("\n=== Cluster Summary ===")
    print(f"{'Cluster':>9}  {'Topic':<14}  {'N':>6}  {'Mean':>7}  {'Median':>8}  {'Std':>6}")
    print("─" * 65)
    grp = data.groupby(["cluster_id", "topic"])["stance"]
    for (cid, topic), g in sorted(grp, key=lambda x: x[0][0]):
        print(f"  #{cid:<7}  {topic:<14}  {len(g):>6,}  "
              f"{g.mean():>+7.3f}  {g.median():>+8.3f}  {g.std():>6.3f}")

    print("\n=== Topic Summary ===")
    for topic in ["Gun Control", "Abortion"]:
        sub = data[data.topic == topic]["stance"]
        if sub.empty:
            continue
        print(f"\n  {topic}  (n={len(sub):,})")
        print(f"    Mean={sub.mean():+.3f}  Median={sub.median():+.3f}  Std={sub.std():.3f}")
        for t in [0.5, 0.7, 0.9]:
            ns = (sub >= t).sum(); no = (sub <= -t).sum()
            print(f"    |score|>={t}  support={ns:,} ({ns/len(sub)*100:.1f}%)  "
                  f"oppose={no:,} ({no/len(sub)*100:.1f}%)")


def delete_comments(ids_to_delete: list[str]):
    ids_set = set(ids_to_delete)
    total_removed = 0

    for fname in sorted(os.listdir(SCORES_DIR)):
        if not fname.endswith(".parquet"):
            continue
        path = os.path.join(SCORES_DIR, fname)
        df = pd.read_parquet(path)
        before = len(df)
        df = df[~df["comment_id"].isin(ids_set)]
        removed = before - len(df)
        if removed > 0:
            df.to_parquet(path, index=False)
            print(f"  {fname}: removed {removed} comment(s)  ({before} → {len(df)})")
            total_removed += removed

    not_found = ids_set - set()
    if total_removed == 0:
        print("  No matching comment_ids found.")
    else:
        print(f"\n  Total removed: {total_removed}")


def main():
    parser = argparse.ArgumentParser(description="Review and filter stance comments")

    # Filter options
    parser.add_argument("--topic", choices=["gun", "abortion"],
                        help="Filter by topic (gun / abortion)")
    parser.add_argument("--cluster", type=int,
                        help="Filter by cluster_id")
    parser.add_argument("--min-stance", type=float, default=-1.0,
                        help="Minimum stance score (default: -1.0)")
    parser.add_argument("--max-stance", type=float, default=1.0,
                        help="Maximum stance score (default: +1.0)")
    parser.add_argument("--order", choices=["asc", "desc"], default="desc",
                        help="Sort order: desc=高分優先, asc=低分優先 (default: desc)")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of comments to show (default: 20)")
    parser.add_argument("--body-width", type=int, default=200,
                        help="Max characters of comment body to show (default: 200)")

    # Actions
    parser.add_argument("--delete", nargs="+", metavar="COMMENT_ID",
                        help="Delete these comment_ids from all Parquet files")
    parser.add_argument("--summary", action="store_true",
                        help="Show cluster / topic summary statistics")

    args = parser.parse_args()

    # ── Delete mode ──────────────────────────────────────────────────────────
    if args.delete:
        print(f"Deleting {len(args.delete)} comment(s): {args.delete}")
        delete_comments(args.delete)
        return

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading data...", end=" ", flush=True)
    data = load_all(load_bodies=True)
    print(f"{len(data):,} comments loaded.")

    # ── Summary mode ─────────────────────────────────────────────────────────
    if args.summary:
        show_summary(data)
        return

    # ── Filter ───────────────────────────────────────────────────────────────
    df = data.copy()

    if args.topic == "gun":
        df = df[df.topic == "Gun Control"]
    elif args.topic == "abortion":
        df = df[df.topic == "Abortion"]

    if args.cluster:
        df = df[df.cluster_id == args.cluster]

    df = df[(df.stance >= args.min_stance) & (df.stance <= args.max_stance)]

    ascending = (args.order == "asc")
    df = df.sort_values("stance", ascending=ascending).reset_index(drop=True)

    if df.empty:
        print("No comments match the filters.")
        return

    print_comments(df, limit=args.limit, body_width=args.body_width)


if __name__ == "__main__":
    main()
