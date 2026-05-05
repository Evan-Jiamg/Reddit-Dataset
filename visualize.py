import zstandard as zstd
import json
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

DATA_PATH = os.path.join(
    os.environ["USERPROFILE"],
    ".cache", "kagglehub", "datasets",
    "i221113hadiyatanveer",
    "the-pushshift-reddit-dataset-submissions",
    "versions", "1", "RC_2019-04.zst"
)
SAMPLE_SIZE = 200_000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ────────────────────────────────
print("Loading data...")
records = []
with open(DATA_PATH, "rb") as fh:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(fh) as reader:
        buffer = b""
        count = 0
        while count < SAMPLE_SIZE:
            chunk = reader.read(2 ** 22)
            if not chunk:
                break
            buffer += chunk
            lines = buffer.split(b"\n")
            buffer = lines[-1]
            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                    count += 1
                    if count >= SAMPLE_SIZE:
                        break
                except json.JSONDecodeError:
                    continue

df = pd.DataFrame(records)
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
df["body_len"] = df["body"].astype(str).str.len()
df["hour"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.hour
df["is_deleted"] = df["body"].isin(["[deleted]", "[removed]", ""])
print(f"Done: {len(df):,} records\n")

sns.set_theme(style="whitegrid", palette="muted")

# ════════════════════════════════════════════
# Chart 1: Score distribution
# ════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Comment Score Distribution", fontsize=15, fontweight="bold")

ax = axes[0]
clipped = df["score"].clip(-50, 200)
ax.hist(clipped, bins=80, color="#4C8BE0", edgecolor="white", linewidth=0.4)
ax.set_xlabel("Score (clipped to -50 ~ 200)")
ax.set_ylabel("Number of Comments")
ax.set_title("Main Distribution")
ax.axvline(df["score"].median(), color="tomato", lw=1.5, linestyle="--",
           label=f"Median = {df['score'].median():.0f}")
ax.axvline(df["score"].mean(), color="gold", lw=1.5, linestyle="--",
           label=f"Mean = {df['score'].mean():.1f}")
ax.legend()

ax = axes[1]
sorted_scores = np.sort(df["score"].dropna())
cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
ax.plot(sorted_scores.clip(-50, 500), cdf, color="#4C8BE0", lw=1.5)
ax.set_xlabel("Score")
ax.set_ylabel("Cumulative Proportion")
ax.set_title("Cumulative Distribution (CDF)")
for p, label in [(0.5, "50%"), (0.9, "90%"), (0.99, "99%")]:
    val = np.percentile(sorted_scores, p * 100)
    ax.axhline(p, color="gray", lw=0.8, linestyle=":")
    ax.axvline(val, color="gray", lw=0.8, linestyle=":")
    ax.annotate(f"{label}\n={val:.0f}", xy=(val, p), fontsize=8, color="dimgray")

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart1_score.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ════════════════════════════════════════════
# Chart 2: Top 25 Subreddits
# ════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
top_sr = df["subreddit"].value_counts().head(25)
colors = sns.color_palette("Blues_r", len(top_sr))
bars = ax.barh(top_sr.index[::-1], top_sr.values[::-1], color=colors[::-1])
for bar, val in zip(bars, top_sr.values[::-1]):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({val/len(df)*100:.1f}%)", va="center", fontsize=8)
ax.set_xlabel("Number of Comments")
ax.set_title("Top 25 Most Active Subreddits (first 200k records)", fontsize=13, fontweight="bold")
ax.set_xlim(0, top_sr.max() * 1.25)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart2_subreddits.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ════════════════════════════════════════════
# Chart 3: Body length distribution
# ════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Comment Length Distribution (body characters)", fontsize=15, fontweight="bold")

valid = df.loc[~df["is_deleted"], "body_len"]

ax = axes[0]
ax.hist(valid.clip(0, 2000), bins=80, color="#52B788", edgecolor="white", linewidth=0.4)
ax.set_xlabel("Characters (clipped to 2000)")
ax.set_ylabel("Number of Comments")
ax.set_title("Length Distribution (non-deleted comments)")
ax.axvline(valid.median(), color="tomato", lw=1.5, linestyle="--",
           label=f"Median = {valid.median():.0f}")
ax.axvline(valid.mean(), color="gold", lw=1.5, linestyle="--",
           label=f"Mean = {valid.mean():.0f}")
ax.legend()

ax = axes[1]
buckets = pd.cut(valid, bins=[0, 50, 100, 200, 500, 1000, 3000, 50000],
                 labels=["<=50", "51-100", "101-200", "201-500", "501-1000", "1001-3000", ">3000"])
bucket_counts = buckets.value_counts().sort_index()
ax.bar(bucket_counts.index.astype(str), bucket_counts.values, color="#52B788", edgecolor="white")
for i, v in enumerate(bucket_counts.values):
    ax.text(i, v + 200, f"{v/len(valid)*100:.1f}%", ha="center", fontsize=9)
ax.set_xlabel("Character Range")
ax.set_ylabel("Number of Comments")
ax.set_title("Distribution by Length Bucket")

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart3_body_length.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ════════════════════════════════════════════
# Chart 4: Missing values
# ════════════════════════════════════════════
null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = ["#E63946" if v > 50 else "#F4A261" if v > 10 else "#A8DADC"
              for v in null_pct.values]
ax.barh(null_pct.index[::-1], null_pct.values[::-1], color=bar_colors[::-1])
for i, (col, val) in enumerate(zip(null_pct.index[::-1], null_pct.values[::-1])):
    ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Missing Rate (%)")
ax.set_title("Missing Values by Column", fontsize=13, fontweight="bold")
ax.set_xlim(0, 115)
legend = [mpatches.Patch(color="#E63946", label=">50%"),
          mpatches.Patch(color="#F4A261", label="10-50%"),
          mpatches.Patch(color="#A8DADC", label="<10%")]
ax.legend(handles=legend, loc="lower right")
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart4_missing.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ════════════════════════════════════════════
# Chart 5: Hourly activity & deletion rate
# ════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Hourly Analysis (UTC) — Sample covers hour 0 only", fontsize=13, fontweight="bold")

hourly = df.groupby("hour").size()
ax = axes[0]
ax.bar(hourly.index, hourly.values, color="#4C8BE0", edgecolor="white")
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Number of Comments")
ax.set_title("Comments per Hour")
ax.set_xticks(range(0, 24))

del_rate = df.groupby("hour")["is_deleted"].mean() * 100
ax = axes[1]
ax.plot(del_rate.index, del_rate.values, marker="o", color="tomato", linewidth=2)
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Deletion Rate (%)")
ax.set_title("Deletion Rate per Hour")
ax.set_xticks(range(0, 24))

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart5_hourly.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ════════════════════════════════════════════
# Chart 6: Score vs body length
# ════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Score vs. Comment Length", fontsize=13, fontweight="bold")

top10_sr = df["subreddit"].value_counts().head(10).index
sample = df[df["subreddit"].isin(top10_sr) & ~df["is_deleted"]].sample(
    min(5000, len(df)), random_state=42)

ax = axes[0]
ax.scatter(sample["body_len"].clip(0, 3000), sample["score"].clip(-20, 200),
           alpha=0.15, s=8, color="#4C8BE0")
ax.set_xlabel("Comment Length (characters)")
ax.set_ylabel("Score")
ax.set_title("Body Length vs. Score")
corr = df[["body_len", "score"]].corr().iloc[0, 1]
ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
        fontsize=10, va="top", color="dimgray")

ax = axes[1]
sr_med = df.groupby("subreddit")["score"].median().sort_values(ascending=False).head(20)
ax.barh(sr_med.index[::-1], sr_med.values[::-1], color="#F4A261", edgecolor="white")
ax.set_xlabel("Median Score")
ax.set_title("Top 20 Subreddits by Median Score")
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart6_score_analysis.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ════════════════════════════════════════════
# Chart 7: Deletion overview
# ════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Comment Deletion Analysis", fontsize=13, fontweight="bold")

ax = axes[0]
del_counts = df["is_deleted"].value_counts()
pie_labels = ["Active", "Deleted / Removed"]
pie_colors = ["#52B788", "#E63946"]
ax.pie(del_counts.values, labels=pie_labels, colors=pie_colors,
       autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
ax.set_title("Overall Deletion Rate")

ax = axes[1]
top20 = df["subreddit"].value_counts().head(20).index
del_by_sr = df[df["subreddit"].isin(top20)].groupby("subreddit")["is_deleted"].mean() * 100
del_by_sr = del_by_sr.sort_values(ascending=False)
bar_colors2 = ["#E63946" if v > 10 else "#F4A261" if v > 5 else "#52B788"
               for v in del_by_sr.values]
ax.barh(del_by_sr.index[::-1], del_by_sr.values[::-1], color=bar_colors2[::-1])
ax.set_xlabel("Deletion Rate (%)")
ax.set_title("Deletion Rate by Subreddit (Top 20 active)")
for i, v in enumerate(del_by_sr.values[::-1]):
    ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlim(0, del_by_sr.max() * 1.3)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart7_deleted.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

print("\nAll charts generated.")
