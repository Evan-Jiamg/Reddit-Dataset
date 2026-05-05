"""
Full streaming EDA for RC_2019-04.zst
Reads the entire file without loading into memory.
Results saved to eda_report.txt and eda_charts/*.png
"""
import zstandard as zstd
import json
import numpy as np
import os
import time
from collections import Counter
from datetime import datetime, timezone
import math

DATA_PATH = os.path.join(
    os.environ["USERPROFILE"],
    ".cache", "kagglehub", "datasets",
    "i221113hadiyatanveer",
    "the-pushshift-reddit-dataset-submissions",
    "versions", "1", "RC_2019-04.zst"
)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(OUTPUT_DIR, "eda_charts")
os.makedirs(CHART_DIR, exist_ok=True)

REPORT_EVERY = 1_000_000   # print progress every N records
COUNTER_LIMIT = 500_000    # max unique keys kept in large Counters
UNIQUE_CAP    = 2_000_000  # stop growing unique-author set after this many

# ── Welford online mean/variance ──────────────
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return
        x = float(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
        if x < self.min_val:
            self.min_val = x
        if x > self.max_val:
            self.max_val = x

    @property
    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self):
        return math.sqrt(self.variance)


# ── Histogram accumulator ─────────────────────
class StreamHistogram:
    def __init__(self, edges):
        self.edges = np.array(edges)
        self.counts = np.zeros(len(edges) - 1, dtype=np.int64)

    def update(self, x):
        if x is None:
            return
        idx = np.searchsorted(self.edges, x, side="right") - 1
        idx = max(0, min(idx, len(self.counts) - 1))
        self.counts[idx] += 1


# ── Accumulators ──────────────────────────────
total            = 0
parse_errors     = 0
null_counts      = Counter()

score_stats      = RunningStats()
score_hist       = StreamHistogram([-1000,-100,-10,-5,-1,0,1,2,3,4,5,10,
                                     20,50,100,200,500,1000,5000,50000])
body_len_stats   = RunningStats()
body_len_hist    = StreamHistogram([0,25,50,75,100,150,200,300,500,
                                    750,1000,2000,5000,50000])

deleted_count    = 0
controversiality_count = 0
gilded_sum       = 0
awards_sum       = 0
awards_n         = 0

subreddit_ctr    = Counter()
hour_ctr         = Counter()
sr_type_ctr      = Counter()
distinguished_ctr= Counter()
unique_authors   = set()
unique_authors_capped = False

bool_counts = {
    "can_gild": 0, "can_mod_post": 0, "collapsed": 0,
    "is_submitter": 0, "locked_true": 0, "no_follow": 0,
    "quarantined": 0, "send_replies": 0, "stickied": 0,
}

sr_score_sum  = Counter()
sr_score_n    = Counter()
sr_del_sum    = Counter()
sr_del_n      = Counter()

TRACKED_FIELDS = [
    "all_awardings","author","author_created_utc","author_flair_background_color",
    "author_flair_css_class","author_flair_richtext","author_flair_template_id",
    "author_flair_text","author_flair_text_color","author_flair_type",
    "author_fullname","author_patreon_flair","body","can_gild","can_mod_post",
    "collapsed","collapsed_reason","controversiality","created_utc","distinguished",
    "edited","gilded","gildings","id","is_submitter","link_id","locked","no_follow",
    "parent_id","permalink","quarantined","removal_reason","retrieved_on","score",
    "send_replies","stickied","subreddit","subreddit_id","subreddit_name_prefixed",
    "subreddit_type","total_awards_received","author_cakeday","archived",
]

# ── Stream the file ───────────────────────────
print(f"[{datetime.now():%H:%M:%S}] Starting full stream of RC_2019-04.zst ...")
print(f"  File size: {os.path.getsize(DATA_PATH)/1e9:.1f} GB (compressed)")
start = time.time()

with open(DATA_PATH, "rb") as fh:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(fh) as reader:
        buffer = b""
        while True:
            chunk = reader.read(2 ** 23)
            if not chunk:
                break
            buffer += chunk
            lines = buffer.split(b"\n")
            buffer = lines[-1]

            for raw in lines[:-1]:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    r = json.loads(raw)
                except Exception:
                    parse_errors += 1
                    continue

                total += 1

                for f in TRACKED_FIELDS:
                    if f not in r or r[f] is None:
                        null_counts[f] += 1

                sc = r.get("score")
                if sc is not None:
                    score_stats.update(sc)
                    score_hist.update(sc)

                body = r.get("body", "")
                is_del = body in ("[deleted]", "[removed]", "")
                if is_del:
                    deleted_count += 1
                else:
                    blen = len(body)
                    body_len_stats.update(blen)
                    body_len_hist.update(blen)

                if r.get("controversiality"):
                    controversiality_count += 1

                g = r.get("gilded", 0)
                if g:
                    gilded_sum += g

                aw = r.get("total_awards_received")
                if aw is not None:
                    awards_sum += aw
                    awards_n += 1

                utc = r.get("created_utc")
                if utc:
                    try:
                        hour_ctr[int(utc) // 3600 % 24] += 1
                    except Exception:
                        pass

                sr = r.get("subreddit", "")
                if sr:
                    subreddit_ctr[sr] += 1
                    sr_del_n[sr] += 1
                    if is_del:
                        sr_del_sum[sr] += 1
                    if sc is not None:
                        sr_score_sum[sr] += sc
                        sr_score_n[sr] += 1

                srt = r.get("subreddit_type", "")
                if srt:
                    sr_type_ctr[srt] += 1

                dist = r.get("distinguished")
                if dist:
                    distinguished_ctr[dist] += 1

                author = r.get("author", "")
                if author and not unique_authors_capped:
                    unique_authors.add(author)
                    if len(unique_authors) >= UNIQUE_CAP:
                        unique_authors_capped = True

                for key in ("can_gild","can_mod_post","collapsed",
                            "is_submitter","no_follow","quarantined",
                            "send_replies","stickied"):
                    if r.get(key):
                        bool_counts[key] += 1
                locked = r.get("locked")
                if locked is True or locked == "True":
                    bool_counts["locked_true"] += 1

                if total % REPORT_EVERY == 0:
                    elapsed = time.time() - start
                    rate = total / elapsed
                    print(f"  [{datetime.now():%H:%M:%S}] {total:>12,} records | "
                          f"{rate/1e3:.0f}k rec/s | {elapsed/60:.1f} min elapsed")

# ── Process remaining buffer ──────────────────
for raw in buffer.split(b"\n"):
    raw = raw.strip()
    if not raw:
        continue
    try:
        r = json.loads(raw)
    except Exception:
        parse_errors += 1
        continue
    total += 1
    sc = r.get("score")
    if sc is not None:
        score_stats.update(sc)
    body = r.get("body", "")
    is_del = body in ("[deleted]", "[removed]", "")
    if is_del:
        deleted_count += 1
    sr = r.get("subreddit", "")
    if sr:
        subreddit_ctr[sr] += 1

elapsed_total = time.time() - start
print(f"\n[{datetime.now():%H:%M:%S}] Stream complete: {total:,} records in "
      f"{elapsed_total/60:.1f} min ({total/elapsed_total/1e3:.0f}k rec/s)\n")

# ── Write text report ─────────────────────────
report_path = os.path.join(OUTPUT_DIR, "eda_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    def w(s=""):
        print(s)
        f.write(s + "\n")

    w("=" * 65)
    w("EDA REPORT — RC_2019-04 (Reddit Comments, April 2019)")
    w(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    w("=" * 65)

    w(f"\n[BASIC INFO]")
    w(f"  Total records   : {total:,}")
    w(f"  Parse errors    : {parse_errors:,}")
    w(f"  Elapsed time    : {elapsed_total/60:.1f} min")

    w(f"\n[SCORE]")
    w(f"  Count  : {score_stats.n:,}")
    w(f"  Mean   : {score_stats.mean:.2f}")
    w(f"  Std    : {score_stats.std:.2f}")
    w(f"  Min    : {score_stats.min_val:.0f}")
    w(f"  Max    : {score_stats.max_val:.0f}")

    w(f"\n[BODY LENGTH (non-deleted)]")
    w(f"  Count  : {body_len_stats.n:,}")
    w(f"  Mean   : {body_len_stats.mean:.1f}")
    w(f"  Std    : {body_len_stats.std:.1f}")
    w(f"  Min    : {body_len_stats.min_val:.0f}")
    w(f"  Max    : {body_len_stats.max_val:.0f}")

    w(f"\n[DELETION]")
    w(f"  Deleted / Removed : {deleted_count:,}  ({deleted_count/total*100:.2f}%)")
    w(f"  Active            : {total-deleted_count:,}  ({(total-deleted_count)/total*100:.2f}%)")

    w(f"\n[CONTROVERSIALITY]")
    w(f"  Controversial     : {controversiality_count:,}  ({controversiality_count/total*100:.2f}%)")

    w(f"\n[GILDED]")
    w(f"  Gilded total      : {gilded_sum:,}")

    w(f"\n[AUTHORS]")
    if unique_authors_capped:
        w(f"  Unique authors    : >{UNIQUE_CAP:,} (cap reached)")
    else:
        w(f"  Unique authors    : {len(unique_authors):,}")

    w(f"\n[SUBREDDIT TYPE]")
    for k, v in sr_type_ctr.most_common():
        w(f"  {k:<20} {v:>12,}  ({v/total*100:.2f}%)")

    w(f"\n[DISTINGUISHED]")
    for k, v in distinguished_ctr.most_common():
        w(f"  {k:<20} {v:>12,}  ({v/total*100:.2f}%)")

    w(f"\n[MISSING VALUES (top 20)]")
    for col, cnt in sorted(null_counts.items(), key=lambda x: -x[1])[:20]:
        w(f"  {col:<35} {cnt:>12,}  ({cnt/total*100:.1f}%)")

    w(f"\n[TOP 50 SUBREDDITS]")
    for sr, cnt in subreddit_ctr.most_common(50):
        del_r = sr_del_sum.get(sr, 0) / sr_del_n.get(sr, 1) * 100
        med_sc = sr_score_sum.get(sr, 0) / sr_score_n.get(sr, 1) if sr_score_n.get(sr, 0) else 0
        w(f"  r/{sr:<30} {cnt:>10,} ({cnt/total*100:.2f}%)  "
          f"del={del_r:.1f}%  avg_score={med_sc:.1f}")

    w(f"\n[HOURLY DISTRIBUTION (UTC)]")
    for h in range(24):
        cnt = hour_ctr.get(h, 0)
        bar = "#" * int(cnt / max(hour_ctr.values()) * 40)
        w(f"  {h:02d}h  {bar:<40}  {cnt:,}")

    w("\n" + "=" * 65)
    w("END OF REPORT")

print(f"Report saved to: {report_path}")

# ── Generate charts ───────────────────────────
print("\nGenerating charts...")
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")

# Chart A: Score histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Score Distribution — Full Dataset ({total/1e6:.1f}M comments)", fontsize=14, fontweight="bold")
ax = axes[0]
edges = score_hist.edges
ax.bar(range(len(score_hist.counts)), score_hist.counts, color="#4C8BE0", edgecolor="white")
ax.set_xticks(range(len(score_hist.counts)))
ax.set_xticklabels([f"{int(e)}" for e in edges[:-1]], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Number of Comments")
ax.set_title("Count by Score Bucket")
ax.set_yscale("log")
ax = axes[1]
pcts = score_hist.counts / score_hist.counts.sum() * 100
ax.barh(range(len(pcts)), pcts[::-1], color="#4C8BE0")
ax.set_yticks(range(len(pcts)))
ax.set_yticklabels([f"{int(edges[i])} ~ {int(edges[i+1])}" for i in range(len(pcts))][::-1], fontsize=7)
ax.set_xlabel("% of Comments")
ax.set_title("Score Bucket Share")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "A_score_dist.png"), dpi=150)
plt.close()

# Chart B: Top 30 subreddits
fig, ax = plt.subplots(figsize=(11, 9))
top30 = subreddit_ctr.most_common(30)
names = [x[0] for x in top30][::-1]
vals  = [x[1] for x in top30][::-1]
colors = sns.color_palette("Blues_r", 30)
bars = ax.barh(names, vals, color=colors[::-1])
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() + max(vals)*0.005, bar.get_y() + bar.get_height()/2,
            f"{v/1e6:.2f}M ({v/total*100:.2f}%)", va="center", fontsize=7.5)
ax.set_xlabel("Number of Comments")
ax.set_title(f"Top 30 Most Active Subreddits — Full Dataset ({total/1e6:.1f}M comments)",
             fontsize=12, fontweight="bold")
ax.set_xlim(0, max(vals) * 1.28)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "B_top_subreddits.png"), dpi=150)
plt.close()

# Chart C: Hourly distribution
fig, ax = plt.subplots(figsize=(12, 4))
hours = list(range(24))
cnts  = [hour_ctr.get(h, 0) for h in hours]
ax.bar(hours, cnts, color="#4C8BE0", edgecolor="white")
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Number of Comments")
ax.set_title("Comments by Hour of Day (UTC) — Full April 2019", fontsize=13, fontweight="bold")
ax.set_xticks(hours)
for h, c in zip(hours, cnts):
    ax.text(h, c + max(cnts)*0.005, f"{c/1e6:.1f}M", ha="center", fontsize=7, rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "C_hourly.png"), dpi=150)
plt.close()

# Chart D: Body length histogram
fig, ax = plt.subplots(figsize=(12, 4))
edges_b = body_len_hist.edges
labels_b = [f"{int(edges_b[i])}-{int(edges_b[i+1])}" for i in range(len(body_len_hist.counts))]
ax.bar(range(len(body_len_hist.counts)), body_len_hist.counts, color="#52B788", edgecolor="white")
ax.set_xticks(range(len(body_len_hist.counts)))
ax.set_xticklabels(labels_b, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Number of Comments")
ax.set_title(f"Comment Length Distribution (non-deleted) — Mean={body_len_stats.mean:.0f} chars",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "D_body_length.png"), dpi=150)
plt.close()

# Chart E: Deletion rate & avg score — Top 30 SRs
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.suptitle("Deletion Rate by Subreddit (Top 30 active)", fontsize=13, fontweight="bold")
top30_names = [x[0] for x in subreddit_ctr.most_common(30)]
del_rates = [(sr_del_sum.get(s,0)/sr_del_n.get(s,1)*100) for s in top30_names]
sorted_pairs = sorted(zip(top30_names, del_rates), key=lambda x: x[1])
sr_sorted, dr_sorted = zip(*sorted_pairs)
ax = axes[0]
bar_colors = ["#E63946" if v > 15 else "#F4A261" if v > 7 else "#52B788" for v in dr_sorted]
ax.barh(sr_sorted, dr_sorted, color=bar_colors)
for i, v in enumerate(dr_sorted):
    ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Deletion Rate (%)")
ax.set_title("Sorted by deletion rate")
ax.set_xlim(0, max(dr_sorted) * 1.25)
avg_scores = [(sr_score_sum.get(s,0)/sr_score_n.get(s,1)) if sr_score_n.get(s,0) else 0
              for s in top30_names]
sorted_sc = sorted(zip(top30_names, avg_scores), key=lambda x: x[1])
sr_sc_sorted, sc_sorted = zip(*sorted_sc)
ax = axes[1]
ax.barh(sr_sc_sorted, sc_sorted, color="#F4A261", edgecolor="white")
ax.set_xlabel("Avg Score")
ax.set_title("Sorted by average score")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "E_deletion_and_score.png"), dpi=150)
plt.close()

# Chart F: Missing values
null_sorted = sorted(null_counts.items(), key=lambda x: -x[1])
cols_m = [x[0] for x in null_sorted]
pcts_m = [x[1]/total*100 for x in null_sorted]
fig, ax = plt.subplots(figsize=(10, 7))
bar_c = ["#E63946" if v > 50 else "#F4A261" if v > 10 else "#A8DADC" for v in pcts_m]
ax.barh(cols_m[::-1], pcts_m[::-1], color=bar_c[::-1])
for i, v in enumerate(pcts_m[::-1]):
    ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Missing Rate (%)")
ax.set_title("Missing Values by Column — Full Dataset", fontsize=13, fontweight="bold")
ax.set_xlim(0, 115)
legend = [mpatches.Patch(color="#E63946", label=">50%"),
          mpatches.Patch(color="#F4A261", label="10-50%"),
          mpatches.Patch(color="#A8DADC", label="<10%")]
ax.legend(handles=legend, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "F_missing.png"), dpi=150)
plt.close()

print(f"\nAll charts saved to: {CHART_DIR}")
print("Done.")
