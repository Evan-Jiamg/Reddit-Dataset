"""
Stage 1: SBERT Topic Clustering
Stream RC_2019-04.zst → filter subreddit → embed with SBERT → cluster with HDBSCAN
Output: clusters_<subreddit>.json

Resumable: embeddings are cached to embeddings_<subreddit>.npy so re-runs skip
the slow encoding step (useful when tuning --min-cluster-size or --umap-dims).
"""

import os
import json
import argparse
import numpy as np
import zstandard as zstd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import hdbscan

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

SBERT_MODEL = "all-MiniLM-L6-v2"
MIN_BODY_LEN = 20
MAX_COMMENTS = 500_000


def stream_subreddit(data_path: str, subreddit: str, max_comments: int,
                     max_per_author: int = 0):
    """Yield (comment_id, body, author, score) for non-deleted comments in target subreddit.

    max_per_author: if > 0, cap each author to this many comments globally (first-seen
    wins).  The per-cluster author dedup in run() still runs afterwards; this is a cheap
    pre-filter that stops prolific authors from dominating the embedding space.
    """
    target = subreddit.lower()
    count = 0
    author_seen: dict[str, int] = {}  # author -> number of comments already yielded
    with open(data_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while count < max_comments:
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
                    if rec.get("subreddit", "").lower() != target:
                        continue
                    body = rec.get("body", "")
                    if body in ("[deleted]", "[removed]", "") or len(body) < MIN_BODY_LEN:
                        continue
                    author = rec.get("author", "[unknown]")
                    if author in ("[deleted]", "[removed]", ""):
                        author = "[unknown]"
                    if max_per_author > 0 and author_seen.get(author, 0) >= max_per_author:
                        continue
                    cid = rec.get("id") or rec.get("name", "")
                    score = int(rec.get("score", 1))
                    yield cid, body, author, score
                    author_seen[author] = author_seen.get(author, 0) + 1
                    count += 1
                    if count >= max_comments:
                        break
    print(f"  Streamed {count:,} comments from r/{subreddit}"
          + (f"  (max_per_author={max_per_author}, unique authors={len(author_seen):,})"
             if max_per_author else ""))


def extract_keywords(texts: list[str], top_n: int = 8) -> str:
    """Return top TF-IDF keywords as a comma-separated string."""
    if len(texts) < 2:
        return texts[0][:80] if texts else ""
    vec = TfidfVectorizer(max_features=5000, stop_words="english",
                          min_df=2, ngram_range=(1, 2))
    try:
        X = vec.fit_transform(texts)
        scores = np.asarray(X.mean(axis=0)).ravel()
        top_idx = scores.argsort()[-top_n:][::-1]
        terms = vec.get_feature_names_out()
        return ", ".join(terms[i] for i in top_idx)
    except ValueError:
        return ""


def run(subreddit: str, min_cluster_size: int, max_comments: int,
        output_dir: str, umap_dims: int, max_per_author: int = 0):
    print(f"\n=== Stage 1: Clustering r/{subreddit} ===")

    cache_dir = output_dir
    emb_cache = os.path.join(cache_dir, f"embeddings_{subreddit}.npy")
    ids_cache = os.path.join(cache_dir, f"ids_{subreddit}.json")

    # 1. Load from cache or stream + embed
    cache_valid = (
        os.path.exists(emb_cache) and
        os.path.exists(ids_cache) and
        "authors" in json.load(open(ids_cache, encoding="utf-8"))
    ) if os.path.exists(ids_cache) else False

    if cache_valid:
        print(f"Loading cached embeddings from {emb_cache} ...")
        embeddings = np.load(emb_cache)
        with open(ids_cache, encoding="utf-8") as f:
            cache_data = json.load(f)
        ids = cache_data["ids"]
        bodies = cache_data["bodies"]
        authors = cache_data["authors"]
        scores = cache_data["scores"]
        print(f"  Loaded {len(ids):,} comments, embeddings shape: {embeddings.shape}")
    else:
        # Stream comments
        print("Streaming comments...")
        ids, bodies, authors, scores = [], [], [], []
        for cid, body, author, score in stream_subreddit(DATA_PATH, subreddit, max_comments,
                                                         max_per_author=max_per_author):
            ids.append(cid)
            bodies.append(body)
            authors.append(author)
            scores.append(score)

        if not ids:
            print("No comments found. Check subreddit name.")
            return

        # Embed with SBERT
        print(f"Embedding {len(bodies):,} comments with {SBERT_MODEL}...")
        sbert = SentenceTransformer(SBERT_MODEL)
        embeddings = sbert.encode(
            bodies,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        print(f"  Embeddings shape: {embeddings.shape}")

        # Save cache
        os.makedirs(cache_dir, exist_ok=True)
        np.save(emb_cache, embeddings)
        with open(ids_cache, "w", encoding="utf-8") as f:
            json.dump({"ids": ids, "bodies": bodies, "authors": authors, "scores": scores},
                      f, ensure_ascii=False)
        print(f"  Cached embeddings → {emb_cache}")

    # 2. Optional UMAP dimensionality reduction
    cluster_input = embeddings
    if umap_dims > 0 and umap_dims < embeddings.shape[1]:
        print(f"Reducing {embeddings.shape[1]}→{umap_dims} dims with UMAP...")

        # Try GPU UMAP (RAPIDS cuml) first, fall back to CPU umap-learn
        cuml_path = "/mnt/NewSSD/CS_project/pypackages"
        _used_gpu = False
        try:
            import sys
            if cuml_path not in sys.path:
                sys.path.insert(0, cuml_path)
            from cuml.manifold import UMAP as cuUMAP
            import cupy as cp
            print("  Using GPU UMAP (cuml)...")
            reducer = cuUMAP(
                n_components=umap_dims,
                n_neighbors=15,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )
            cluster_input = reducer.fit_transform(embeddings)
            if hasattr(cluster_input, "get"):          # cupy array → numpy
                cluster_input = cluster_input.get()
            _used_gpu = True
            print(f"  Reduced shape (GPU): {cluster_input.shape}")
        except Exception as e:
            if _used_gpu:
                raise
            print(f"  cuml not available ({e}), falling back to CPU umap-learn...")
            try:
                import umap
            except ImportError:
                print("  umap-learn not installed. Run: pip install umap-learn")
                print("  Skipping UMAP, using full embeddings.")
            else:
                reducer = umap.UMAP(
                    n_components=umap_dims,
                    n_neighbors=15,
                    min_dist=0.0,
                    metric="cosine",
                    random_state=42,
                    low_memory=True,
                )
                cluster_input = reducer.fit_transform(embeddings)
                print(f"  Reduced shape (CPU): {cluster_input.shape}")

    # 3. HDBSCAN clustering — try GPU (cuml) first, fall back to CPU
    print(f"Clustering (min_cluster_size={min_cluster_size})...")
    _used_gpu_hdbscan = False
    try:
        cuml_path = "/mnt/NewSSD/CS_project/reddit_env/lib/python3.12/site-packages"
        import sys
        if cuml_path not in sys.path:
            sys.path.insert(0, cuml_path)
        from cuml.cluster import HDBSCAN as cuHDBSCAN
        import cupy as cp
        print("  Using GPU HDBSCAN (cuml)...")
        gpu_clusterer = cuHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = gpu_clusterer.fit_predict(cluster_input.astype("float32"))
        if hasattr(labels, "get"):
            labels = labels.get()
        import numpy as _np
        labels = _np.asarray(labels)
        _used_gpu_hdbscan = True
        print("  GPU HDBSCAN done.")
    except Exception as e:
        if _used_gpu_hdbscan:
            raise
        print(f"  cuml HDBSCAN unavailable ({e}), using CPU hdbscan...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(cluster_input)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int((labels == -1).sum())
    print(f"  Found {n_clusters} clusters, {noise_count:,} noise points")

    # 4. Build cluster objects
    clusters = []
    for cid_val in sorted(set(labels)):
        if cid_val == -1:
            continue
        mask = labels == cid_val
        indices = [i for i, m in enumerate(mask) if m]

        # Deduplicate: per author keep only the highest-score comment
        best: dict[str, tuple[int, int]] = {}  # author -> (index, score)
        for i in indices:
            author = authors[i]
            score = scores[i]
            if author not in best or score > best[author][1]:
                best[author] = (i, score)

        dedup_indices = [v[0] for v in best.values()]
        cluster_ids = [ids[i] for i in dedup_indices]
        cluster_bodies = [bodies[i] for i in dedup_indices]

        keywords = extract_keywords(cluster_bodies)
        kw_list = [k.strip() for k in keywords.split(",")]
        topic_label = " | ".join(kw_list[:5])
        topic_description = f"Discussion about: {keywords}"

        clusters.append({
            "cluster_id": int(cid_val),
            "topic_label": topic_label,
            "topic_description": topic_description,
            "comment_ids": cluster_ids,
            "size": len(cluster_ids),
            "size_before_dedup": int(mask.sum()),
        })

    clusters.sort(key=lambda x: x["size"], reverse=True)

    # 5. Print summary
    print(f"\nTop 20 clusters:")
    print(f"{'#':>4}  {'Size':>6}  Topic")
    print("-" * 70)
    for c in clusters[:20]:
        print(f"{c['cluster_id']:>4}  {c['size']:>6}  {c['topic_label'][:55]}")

    # 6. Save output
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"clusters_{subreddit}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(clusters)} clusters → {out_path}")

    meta = {
        "subreddit": subreddit,
        "total_comments": len(ids),
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "min_cluster_size": min_cluster_size,
        "umap_dims": umap_dims,
        "sbert_model": SBERT_MODEL,
    }
    meta_path = os.path.join(output_dir, f"clusters_{subreddit}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: SBERT + HDBSCAN topic clustering")
    parser.add_argument("--subreddit", default="politics")
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--max-comments", type=int, default=MAX_COMMENTS)
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--umap-dims", type=int, default=50,
                        help="UMAP target dimensions before HDBSCAN (0 = skip UMAP)")
    parser.add_argument("--max-per-author", type=int, default=0,
                        help="Global per-author cap at stream time (0 = no limit). "
                             "Complements the per-cluster author dedup done after HDBSCAN.")
    args = parser.parse_args()

    run(
        subreddit=args.subreddit,
        min_cluster_size=args.min_cluster_size,
        max_comments=args.max_comments,
        output_dir=args.output_dir,
        umap_dims=args.umap_dims,
        max_per_author=args.max_per_author,
    )
