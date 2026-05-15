"""
visualize_stance.py
───────────────────
Read stance_scores/*.parquet → produce 3 paper-quality charts:

  Chart 1: KDE distribution — both topics on one axis, stratification bins shown
  Chart 2: Per-cluster horizontal box plot — all clusters, sorted by median
  Chart 3: Extreme-stance line chart — % beyond threshold (0.5/0.7/0.8/0.9)

Design: single-hue blue palette, Arial 9pt, 6.5" wide, B&W-printable.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCORES_DIR = os.path.join(SCRIPT_DIR, "stance_scores")
CHART_DIR  = os.path.join(SCRIPT_DIR, "stance_charts")
os.makedirs(CHART_DIR, exist_ok=True)

GUN_IDS  = {775, 708, 750, 725, 705, 456}
ABRT_IDS = {132, 131}
THRESHOLDS = [0.5, 0.7, 0.8, 0.9]
BIN_THR    = [-0.75, -0.25, 0.25, 0.75]   # stratification thresholds

# ── Palette (single-hue blues) ────────────────────────────────────────────────
C_GUN  = "#08306b"   # dark blue  — Gun Control
C_ABRT = "#4292c6"   # mid blue   — Abortion
LS_GUN  = "solid"
LS_ABRT = (0, (6, 2))   # long-dash

HATCH_GUN  = ""
HATCH_ABRT = "////"

# ── Paper rcParams ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "legend.fontsize":    8,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.7,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "grid.linestyle":     ":",
    "lines.linewidth":    1.5,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.75",
    "legend.borderpad":   0.4,
    "legend.handlelength": 2.0,
})

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading parquet files...")
frames = []
for fname in sorted(os.listdir(SCORES_DIR)):
    if not fname.endswith(".parquet"):
        continue
    cid = int(fname.replace("cluster_", "").replace(".parquet", ""))
    if cid not in GUN_IDS | ABRT_IDS:
        continue
    df = pd.read_parquet(os.path.join(SCORES_DIR, fname))
    df["topic"]      = "Gun Control" if cid in GUN_IDS else "Abortion"
    df["cluster_id"] = cid
    frames.append(df)

data   = pd.concat(frames, ignore_index=True)
gun    = data[data.topic == "Gun Control"]["stance"].values
abort  = data[data.topic == "Abortion"]["stance"].values
print(f"  Gun Control: n={len(gun):,}   Abortion: n={len(abort):,}")

# ═════════════════════════════════════════════════════════════════════════════
# Chart 1 — KDE distribution, both topics, with stratification bins
# ═════════════════════════════════════════════════════════════════════════════
print("Chart 1: stance distribution...")

x = np.linspace(-1.05, 1.05, 400)
kde_gun  = gaussian_kde(gun,  bw_method="scott")(x)
kde_abrt = gaussian_kde(abort, bw_method="scott")(x)
y_max = max(kde_gun.max(), kde_abrt.max())

fig, ax = plt.subplots(figsize=(6.5, 2.6))

ax.plot(x, kde_gun,  color=C_GUN,  ls=LS_GUN,  lw=1.5,
        label=f"Gun Control ($n$={len(gun):,})")
ax.fill_between(x, kde_gun,  alpha=0.10, color=C_GUN)

ax.plot(x, kde_abrt, color=C_ABRT, ls=LS_ABRT, lw=1.5,
        label=f"Abortion ($n$={len(abort):,})")
ax.fill_between(x, kde_abrt, alpha=0.10, color=C_ABRT)

# Stratification bin boundaries
for thr in BIN_THR:
    ax.axvline(thr, color="0.55", lw=0.6, ls=":", zorder=0)

# Mean lines
ax.axvline(gun.mean(),   color=C_GUN,  lw=0.9, ls=(0, (3, 2)),
           label=f"Mean gun = {gun.mean():+.2f}")
ax.axvline(abort.mean(), color=C_ABRT, lw=0.9, ls=(0, (3, 2)),
           label=f"Mean abortion = {abort.mean():+.2f}")

# Bin labels at top
bin_centers = [-0.875, -0.5, 0.0, 0.5, 0.875]
bin_texts   = ["Strongly\nOppose", "Oppose", "Neutral", "Support", "Strongly\nSupport"]
for cx, txt in zip(bin_centers, bin_texts):
    ax.text(cx, y_max * 1.13, txt,
            ha="center", va="top", fontsize=6, color="0.45")

ax.set_xlim(-1.05, 1.05)
ax.set_ylim(0, y_max * 1.28)
ax.set_xlabel("RoBERTa stance score  ($-1$ = strongly oppose,  $+1$ = strongly support)")
ax.set_ylabel("Density")
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))

ax.legend(loc="upper left", ncol=2, fontsize=7.5)
fig.tight_layout()
p = os.path.join(CHART_DIR, "chart1_distribution.png")
fig.savefig(p)
plt.close(fig)
print(f"  Saved: {p}")

# ═════════════════════════════════════════════════════════════════════════════
# Chart 2 — Per-cluster horizontal box plot
# ═════════════════════════════════════════════════════════════════════════════
print("Chart 2: per-cluster box plot...")

# Build cluster table sorted by median
cluster_info = []
for cid in sorted(GUN_IDS | ABRT_IDS):
    sub = data[data.cluster_id == cid]["stance"].values
    if len(sub) == 0:
        continue
    topic = "Gun Control" if cid in GUN_IDS else "Abortion"
    cluster_info.append(dict(
        cid=cid, topic=topic,
        label=f"{'GC' if topic == 'Gun Control' else 'AB'}-{cid}  ($n$={len(sub):,})",
        data=sub,
        median=np.median(sub),
    ))
cluster_info.sort(key=lambda d: d["median"])

n_cl = len(cluster_info)
fig, ax = plt.subplots(figsize=(6.5, 0.45 * n_cl + 0.7))

for i, ci in enumerate(cluster_info):
    color = C_GUN if ci["topic"] == "Gun Control" else C_ABRT
    q1, med, q3 = np.percentile(ci["data"], [25, 50, 75])
    iqr = q3 - q1
    lo  = max(ci["data"].min(), q1 - 1.5 * iqr)
    hi  = min(ci["data"].max(), q3 + 1.5 * iqr)

    # whiskers
    ax.plot([lo, hi], [i, i], color=color, lw=0.8, zorder=2)
    # box
    ax.barh(i, q3 - q1, left=q1, height=0.45,
            color=color, alpha=0.30,
            hatch=HATCH_GUN if ci["topic"] == "Gun Control" else HATCH_ABRT,
            edgecolor=color, linewidth=0.6, zorder=3)
    # median
    ax.plot([med, med], [i - 0.225, i + 0.225], color=color, lw=1.5, zorder=4)
    # mean dot
    ax.scatter(ci["data"].mean(), i, color="white", edgecolors=color,
               s=18, lw=0.8, zorder=5)

ax.axvline(0, color="0.50", lw=0.6, ls=":", zorder=1)
ax.set_yticks(range(n_cl))
ax.set_yticklabels([ci["label"] for ci in cluster_info], fontsize=7.5)
ax.set_xlabel("Stance score")
ax.set_xlim(-1.1, 1.1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))

# Legend patches
import matplotlib.patches as mpatches
handles = [
    mpatches.Patch(facecolor=C_GUN,  alpha=0.3, edgecolor=C_GUN,  label="Gun Control"),
    mpatches.Patch(facecolor=C_ABRT, alpha=0.3, edgecolor=C_ABRT,
                   hatch=HATCH_ABRT, label="Abortion"),
    plt.Line2D([0], [0], color="0.4", lw=1.5, label="Median"),
    plt.Line2D([0], [0], marker="o", color="w", markeredgecolor="0.4",
               markersize=4, label="Mean"),
]
ax.legend(handles=handles, loc="lower right", fontsize=7.5, ncol=2)

fig.tight_layout()
p = os.path.join(CHART_DIR, "chart2_per_cluster.png")
fig.savefig(p)
plt.close(fig)
print(f"  Saved: {p}")

# ═════════════════════════════════════════════════════════════════════════════
# Chart 3 — Extreme-stance line chart
# ═════════════════════════════════════════════════════════════════════════════
print("Chart 3: extreme-stance proportions...")

records = {}
for topic, arr in [("Gun Control", gun), ("Abortion", abort)]:
    pct_s = [(arr >=  t).mean() * 100 for t in THRESHOLDS]
    pct_o = [(arr <= -t).mean() * 100 for t in THRESHOLDS]
    records[topic] = (pct_s, pct_o)

fig, ax = plt.subplots(figsize=(6.5, 2.4))

x_thr = np.array(THRESHOLDS)

# Gun Control
pct_s, pct_o = records["Gun Control"]
ax.plot(x_thr, pct_s, color=C_GUN, ls=LS_GUN,       lw=1.5, marker="o", ms=4,
        label="Gun Control — Support (≥ +t)")
ax.plot(x_thr, pct_o, color=C_GUN, ls=(0, (3,1,1,1)), lw=1.5, marker="s", ms=4,
        label="Gun Control — Oppose  (≤ −t)")

# Abortion
pct_s, pct_o = records["Abortion"]
ax.plot(x_thr, pct_s, color=C_ABRT, ls=LS_ABRT,       lw=1.5, marker="o", ms=4,
        label="Abortion — Support (≥ +t)")
ax.plot(x_thr, pct_o, color=C_ABRT, ls=(0, (3,1,1,1)), lw=1.5, marker="s", ms=4,
        label="Abortion — Oppose  (≤ −t)")

# Value labels at each point
for topic, arr, c in [("Gun Control", gun, C_GUN), ("Abortion", abort, C_ABRT)]:
    for t, ps, po in zip(THRESHOLDS,
                         [(arr >= t).mean()*100 for t in THRESHOLDS],
                         [(arr <=-t).mean()*100 for t in THRESHOLDS]):
        ax.text(t, ps + 0.8, f"{ps:.1f}%", ha="center", va="bottom",
                fontsize=6, color=c)
        ax.text(t, po + 0.8, f"{po:.1f}%", ha="center", va="bottom",
                fontsize=6, color=c)

ax.set_xlabel(r"Threshold $t$  (proportion of comments with $|$score$| \geq t$)")
ax.set_ylabel("% of comments")
ax.set_xticks(THRESHOLDS)
ax.set_xlim(0.45, 0.95)
ax.set_ylim(0)

fig.legend(ncol=2, loc="lower center",
           bbox_to_anchor=(0.5, 0.0), fontsize=7.5)
fig.subplots_adjust(bottom=0.32)
p = os.path.join(CHART_DIR, "chart3_extremes.png")
fig.savefig(p)
plt.close(fig)
print(f"  Saved: {p}")

print(f"\nAll charts saved to: {CHART_DIR}")
