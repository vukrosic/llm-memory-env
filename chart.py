"""Draw every chart in charts/ from the tables in RESULTS.md. python3 chart.py"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); TEXT = open(os.path.join(HERE, "RESULTS.md")).read(); os.makedirs(os.path.join(HERE, "charts"), exist_ok=True)
INK, MUTED, GRID, TEAL, CORAL, PURPLE, GRAY = "#26262b", "#6b6b73", "#e6e6ea", "#14907f", "#e0603f", "#7c5cc4", "#8c8c94"
BASE, CONTROL = {"token", "engram", "moe", "ve", "mome", "stem", "bigram", "tngram"}, {"engmoe", "vemoe", "momemoe"}
OFF_AXIS = {"nmtrip1", "nmtrip5", "nplain", "bigram6"}  # new-wording loss above 4.5; RESULTS.md says so under the chart
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def rows(after):
    """Rows of the first table after a heading or marker, as lists of cell strings."""
    part = TEXT[TEXT.index(after):].split("\n"); out = []
    for l in part[1:]:
        if l.startswith("|"): out.append([c.strip() for c in l.strip("|").split("|")])
        elif out: break
    return out[2:]


pts = {r[0]: (float(r[7]), float(r[5])) for r in rows("<summary>5m,") if r[0] not in OFF_AXIS}
fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
NAMED = {"magree", "mexp8", "mapair", "mbuck64"}
ax.scatter([pts[m][0] for m in NAMED], [pts[m][1] for m in NAMED], s=70, c=CORAL, linewidths=1, edgecolors="white", zorder=3)
groups = [("New designs", [m for m in pts if m not in BASE | CONTROL | NAMED | {"plain"}], CORAL, 34, 0.55), ("Published modules", [m for m in pts if m in BASE], TEAL, 70, 1), ("Published module + MoE", [m for m in pts if m in CONTROL], PURPLE, 70, 1), ("No memory", ["plain"], GRAY, 70, 1)]
for name, ms, col, size, a in groups:
    ax.scatter([pts[m][0] for m in ms], [pts[m][1] for m in ms], s=size, c=col, alpha=a, linewidths=1, edgecolors="white", label=name, zorder=3)
LABEL = {"magree": (9, -9), "mbuck64": (8, 0), "mapair": (8, 4), "mexp8": (8, -4), "engram": (8, 3), "ve": (-8, -3), "mome": (-10, 5), "momemoe": (-14, -14), "engmoe": (8, 2), "plain": (8, 3), "bigram": (8, 4), "token": (8, 3), "stem": (8, -10), "moe": (-8, 3), "tngram": (8, -9), "vemoe": (8, -4)}
for m, off in LABEL.items():
    ax.annotate(m, pts[m], xytext=off, textcoords="offset points", fontsize=9.5, color=INK, ha="right" if off[0] < 0 else "left", zorder=4)
ax.set_xlim(2.9, 6.0); ax.set_ylim(3.7, 0.45); ax.grid(True, color=GRID, lw=.8, zorder=0); ax.set_axisbelow(True)
ax.annotate("better on both", xy=(5.92, .53), xytext=(5.15, .95), fontsize=10, color=MUTED, ha="left", arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
ax.set_xlabel("Relation gap (higher is better)"); ax.set_ylabel("New-wording fact loss (lower is better, axis flipped)")
ax.set_title("At 5m, no run matches the best baseline on both recalling a fact\nin a new wording and holding it back when the question changes", loc="left", fontsize=13, color=INK, pad=12)
ax.legend(frameon=False, loc="lower left", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "charts", "tradeoff-5m.png")); plt.close(fig)

t = rows("`magree` at 5m:")
fig, ax = plt.subplots(figsize=(9, 3.9), dpi=160)
for i, r in enumerate(t):
    own, wrong = float(r[1]), float(r[2]); y = len(t) - 1 - i
    ax.plot([wrong, own], [y, y], color="#c9c9d0", lw=2, zorder=1)
    ax.scatter([own], [y], s=75, c=TEAL, edgecolors="white", linewidths=1.5, zorder=3, label="Right answer, own question" if i == 0 else None)
    ax.scatter([wrong], [y], s=75, c=CORAL, edgecolors="white", linewidths=1.5, zorder=3, label="Same answer, another relation's question" if i == 0 else None)
    ax.annotate(f"gap {float(r[3]):.2f}", ((own + wrong) / 2, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9.5, color=INK)
ax.set_yticks(range(len(t))); ax.set_yticklabels([r[0] for r in t][::-1], color=INK); ax.set_xlim(-10, 0); ax.set_ylim(-.6, len(t) - .3)
ax.grid(True, axis="x", color=GRID, lw=.8, zorder=0); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Log-probability of the stored answer (higher means more likely)"); ax.set_ylabel("Switched off at test time", color=MUTED)
fig.suptitle("Without magree's gated table the gap rises, because that table lifts the wrong answer more", x=.01, ha="left", fontsize=11.5, color=INK)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "charts", "switch-off.png")); plt.close(fig)
