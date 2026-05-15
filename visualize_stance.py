"""
visualize_stance.py
───────────────────
Read stance_scores/*.parquet → produce 4 charts + console summary:

  Chart 1: KDE + histogram  (gun vs abortion, side-by-side)
  Chart 2: Per-cluster box plot
  Chart 3: Extreme-value bar chart (at |stance| ≥ 0.5 / 0.7 / 0.8 / 0.9)
  Chart 4: Sample extreme comments (top-10 support + oppose per topic)
"""

import os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCORES_DIR = "/mnt/NewSSD/CS_project/Reddit-Dataset/stance_scores"
CHART_DIR  = os.path.join(SCRIPT_DIR, "stance_charts")
IDS_CACHE  = "/mnt/NewSSD/CS_project/Reddit-Dataset/data/ids_politics.json"

GUN_IDS      = {90, 220, 344}
ABORTION_IDS = {160}
THRESHOLDS   = [0.5, 0.7, 0.8, 0.9]

os.makedirs(CHART_DIR, exist_ok=True)

# ── 1. Load all parquets ──────────────────────────────────────────────────────
print("Loading Parquet files...")
frames = []
for fname in sorted(os.listdir(SCORES_DIR)):
    if not fname.endswith(".parquet"):
        continue
    cid = int(fname.replace("cluster_", "").replace(".parquet", ""))
    df  = pd.read_parquet(os.path.join(SCORES_DIR, fname))
    if cid in GUN_IDS:
        df["topic"] = "Gun Control"
    elif cid in ABORTION_IDS:
        df["topic"] = "Abortion"
    else:
        continue
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
n_gun  = (data.topic == "Gun Control").sum()
n_abrt = (data.topic == "Abortion").sum()
print(f"  {len(data):,} comments total  ({n_gun:,} gun, {n_abrt:,} abortion)")

# cluster short labels  (sorted by size, largest first)
CLUSTER_META = {
    90: ("Gun Control", 89),
    160: ("Abortion", 1088),
    220: ("Gun Control", 311),
    344: ("Gun Control", 2341),
}
def clabel(cid):
    topic, n = CLUSTER_META[cid]
    short = topic.split()[0]  # "Gun" / "Abortion"
    return f"{short} #{cid}\n(n={n:,})"

data["cluster_label"] = data["cluster_id"].map(clabel)

COLORS = {"Gun Control": "#E74C3C", "Abortion": "#2980B9"}

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1 — KDE + histogram
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 1: distributions...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, topic in zip(axes, ["Gun Control", "Abortion"]):
    sub   = data[data.topic == topic]["stance"]
    color = COLORS[topic]

    ax.hist(sub, bins=80, density=True, alpha=0.30,
            color=color, edgecolor="none", zorder=1)
    sub.plot.kde(ax=ax, color=color, lw=2.5, zorder=3)

    mean_val = sub.mean()
    ax.axvline(mean_val, color="black", ls="--", lw=1.8,
               label=f"Mean = {mean_val:+.3f}", zorder=4)
    ax.axvline(0, color="#555555", ls=":", lw=1, zorder=2)

    # shade extreme regions
    for t, alpha in [(0.7, 0.10), (0.9, 0.18)]:
        ax.axvspan( t,  1.05, alpha=alpha, color=color,  zorder=0)
        ax.axvspan(-1.05, -t, alpha=alpha, color="#7F8C8D", zorder=0)

    # annotate extreme % at |0.9|
    pct_s = (sub >= 0.9).mean() * 100
    pct_o = (sub <= -0.9).mean() * 100
    ax.text(0.96, 0.97, f"≥+0.9: {pct_s:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=color)
    ax.text(0.04, 0.97, f"≤−0.9: {pct_o:.1f}%",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#7F8C8D")

    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Stance score  (−1 = oppose, +1 = support)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{topic}  (n={len(sub):,})", fontsize=12,
                 fontweight="bold", color=color)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

plt.suptitle("Stance Score Distributions — r/politics (April 2019)",
             fontsize=14, y=1.01)
plt.tight_layout()
p = os.path.join(CHART_DIR, "chart1_distribution.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Per-cluster box plot
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 2: per-cluster box plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                         gridspec_kw={"width_ratios": [6, 3]})

for ax, topic, color in [
    (axes[0], "Gun Control", "#E74C3C"),
    (axes[1], "Abortion",    "#2980B9"),
]:
    sub   = data[data.topic == topic]
    order = (sub.groupby("cluster_label")["stance"]
               .median().sort_values(ascending=False).index.tolist())
    groups = [sub[sub.cluster_label == lbl]["stance"].values for lbl in order]

    bp = ax.boxplot(
        groups, vert=True, patch_artist=True, widths=0.5,
        medianprops=dict(color="black", lw=2.5),
        boxprops=dict(facecolor=color, alpha=0.45),
        whiskerprops=dict(lw=1.2, color="#555"),
        capprops=dict(lw=1.2, color="#555"),
        flierprops=dict(marker=".", ms=3, alpha=0.25, color=color),
    )
    # overlay mean dots
    means = [g.mean() for g in groups]
    ax.scatter(range(1, len(order)+1), means, zorder=5,
               color="white", edgecolors=color, s=40, lw=1.5, label="Mean")

    ax.set_xticks(range(1, len(order)+1))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax.axhline(0, color="#777", ls=":", lw=1)
    ax.set_ylabel("Stance score", fontsize=10)
    ax.set_ylim(-1.15, 1.15)
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(topic, fontsize=12, fontweight="bold", color=color)
    ax.legend(fontsize=9)

plt.suptitle("Per-Cluster Stance Distribution\n"
             "Box = IQR, Whiskers = 1.5×IQR, Line = Median, Dot = Mean",
             fontsize=12, y=1.02)
plt.tight_layout()
p = os.path.join(CHART_DIR, "chart2_per_cluster.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Extreme-value bar chart
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 3: extreme values...")
records = []
for topic in ["Gun Control", "Abortion"]:
    sub = data[data.topic == topic]["stance"]
    n   = len(sub)
    for t in THRESHOLDS:
        records.append(dict(
            topic=topic, threshold=t,
            n_support=int((sub >=  t).sum()),
            n_oppose =int((sub <= -t).sum()),
            pct_s=(sub >=  t).mean() * 100,
            pct_o=(sub <= -t).mean() * 100,
            n_total=n,
        ))
ext_df = pd.DataFrame(records)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(THRESHOLDS))
w = 0.35

for ax, topic, color in [(axes[0], "Gun Control", "#E74C3C"),
                          (axes[1], "Abortion",    "#2980B9")]:
    sub = ext_df[ext_df.topic == topic].reset_index(drop=True)
    b1 = ax.bar(x - w/2, sub["pct_s"], w,
                label="Support (score ≥ +threshold)", color=color, alpha=0.82)
    b2 = ax.bar(x + w/2, sub["pct_o"], w,
                label="Oppose  (score ≤ −threshold)", color="#7F8C8D", alpha=0.82)

    for i, row in sub.iterrows():
        ax.text(i - w/2, row.pct_s + 0.4,
                f"{row.pct_s:.1f}%\n({row.n_support:,})",
                ha="center", va="bottom", fontsize=7.5)
        ax.text(i + w/2, row.pct_o + 0.4,
                f"{row.pct_o:.1f}%\n({row.n_oppose:,})",
                ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"|score| ≥ {t}" for t in THRESHOLDS], fontsize=9)
    ax.set_ylabel("% of all comments in topic", fontsize=10)
    ax.set_title(f"{topic}  (total n={sub.n_total.iloc[0]:,})",
                 fontsize=12, fontweight="bold", color=color)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

plt.suptitle("How Many Comments Hold Extreme Stances?", fontsize=13, y=1.01)
plt.tight_layout()
p = os.path.join(CHART_DIR, "chart3_extremes.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4 — Sample extreme comments  (load body text from cache)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 4: extreme comment samples...")

body_map = {}
if os.path.exists(IDS_CACHE):
    print(f"  Loading comment bodies from {IDS_CACHE} ...")
    with open(IDS_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    body_map = dict(zip(cache["ids"], cache["bodies"]))
    print(f"  Loaded {len(body_map):,} bodies")
else:
    print(f"  WARNING: {IDS_CACHE} not found — skipping body text in Chart 4")

SAMPLE_N = 8  # top N per direction per topic

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor("#F8F9FA")

for row_idx, topic in enumerate(["Gun Control", "Abortion"]):
    color = COLORS[topic]
    sub   = data[data.topic == topic].copy()
    sub["body"] = sub["comment_id"].map(body_map) if body_map else "—"

    for col_idx, (direction, sign) in enumerate([("Most Supportive", 1),
                                                  ("Most Opposed",   -1)]):
        ax = axes[row_idx][col_idx]
        ax.set_facecolor("#FFFFFF")
        ax.axis("off")

        if sign == 1:
            sample = sub.nlargest(SAMPLE_N, "stance")
        else:
            sample = sub.nsmallest(SAMPLE_N, "stance")

        title_color = color if sign == 1 else "#555555"
        ax.set_title(f"{topic} — {direction} (top {SAMPLE_N})",
                     fontsize=12, fontweight="bold", color=title_color,
                     pad=8)

        y = 0.96
        for _, r in sample.iterrows():
            body = str(r.get("body", "—"))
            if body and body != "nan":
                # wrap to ~90 chars
                words = body.split()
                lines, line = [], []
                for w in words:
                    line.append(w)
                    if len(" ".join(line)) > 88:
                        lines.append(" ".join(line))
                        line = []
                if line:
                    lines.append(" ".join(line))
                snippet = "\n".join(lines[:3])
                if len(lines) > 3:
                    snippet += "…"
            else:
                snippet = "(body not available)"

            score_str = f"[{r.stance:+.3f}]"
            ax.text(0.01, y, score_str, transform=ax.transAxes,
                    fontsize=9, fontweight="bold",
                    color=title_color, va="top", family="monospace")
            ax.text(0.12, y, snippet, transform=ax.transAxes,
                    fontsize=8, va="top", wrap=True,
                    color="#222222", linespacing=1.4)
            y -= (snippet.count("\n") + 1) * 0.055 + 0.015
            ax.plot([0.01, 0.99], [y + 0.005, y + 0.005],
                    transform=ax.transAxes, color="#DDDDDD", lw=0.7,
                    clip_on=False)

plt.suptitle("Extreme-Stance Comment Samples\n"
             "(scores closest to +1.0 and −1.0)",
             fontsize=14, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
p = os.path.join(CHART_DIR, "chart4_samples.png")
plt.savefig(p, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Console summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("STANCE SUMMARY")
print("═" * 65)
for topic in ["Gun Control", "Abortion"]:
    sub = data[data.topic == topic]["stance"]
    print(f"\n  {topic}  (n={len(sub):,})")
    print(f"    Mean   = {sub.mean():+.4f}")
    print(f"    Median = {sub.median():+.4f}")
    print(f"    Std    = {sub.std():.4f}")
    print(f"    Min    = {sub.min():+.4f}  Max = {sub.max():+.4f}")
    print(f"    {'Threshold':>12}  {'Support':>10}  {'%':>6}  {'Oppose':>10}  {'%':>6}")
    for t in THRESHOLDS:
        ns = int((sub >=  t).sum()); ps = ns/len(sub)*100
        no = int((sub <= -t).sum()); po = no/len(sub)*100
        print(f"    {f'|score|>={t}':>12}  {ns:>10,}  {ps:>5.1f}%  {no:>10,}  {po:>5.1f}%")

print("\n" + "═" * 65)
print(f"All charts saved to: {CHART_DIR}")
