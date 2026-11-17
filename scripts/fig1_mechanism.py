"""Fig. 1 schematic: how the recording-boundary cue arises (vector PDF + PNG)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "hatch.linewidth": 0.5})
EEG, ZERO, MIRROR, LAB, STAT = "#2b5c8a", "#c0392b", "#2a9d8f", "#f2c14e", "#dfe8f1"
rng = np.random.default_rng(3)

def trace(ax, x0, x1, y, amp=1.1, color=EEG, seed=None, n_per=40, data=None):
    n = max(int((x1 - x0) * n_per), 2)
    if data is None:
        r = np.random.default_rng(seed)
        t = np.linspace(0, 1, n)
        data = (0.55 * np.sin(2 * np.pi * (x1 - x0) * 0.45 * t + r.uniform(0, 6))
                + np.convolve(r.normal(0, 1, n + 8), np.ones(9) / 9, "valid")[:n] * 1.4)
    ax.plot(np.linspace(x0, x1, len(data)), y + amp * data / (np.abs(data).max() + 1e-9), color=color, lw=0.55, solid_capstyle="butt")
    return data

def zeros(ax, x0, x1, y, h=1.5):
    ax.add_patch(Rectangle((x0, y - h), x1 - x0, 2 * h, fc="#fbe9e7", ec="none", hatch="////", zorder=0))
    ax.plot([x0, x1], [y, y], color=ZERO, lw=0.9)

def window(ax, x0, x1, y, h=1.9, color="black", ls="-"):
    ax.add_patch(Rectangle((x0, y - h), x1 - x0, 2 * h, fill=False, ec=color, lw=0.8, ls=ls, zorder=5))

def span(ax, x0, x1, y, text):
    ax.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<->", lw=0.5, color="0.35", shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y - 0.35, text, ha="center", va="top", color="0.35", fontsize=5.8)

fig, ax = plt.subplots(figsize=(7.16, 2.55))
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

# ---------- (a) label propagation sets file length ----------
ax.text(0.5, 29.3, "(a) Labelling sets the recording length", fontweight="bold", va="top", fontsize=6.8)
ax.add_patch(Rectangle((5, 19.2), 22, 5.6, fc=STAT, ec="none", zorder=0))
ax.text(16, 25.3, "stationary period", ha="center", fontsize=5.8, color="0.3")
trace(ax, 1, 31, 22, seed=1)
for k, xs in enumerate(np.arange(6, 27, 2.8)):
    ax.add_patch(Rectangle((xs, 19.9), 1.4, 0.45, fc=LAB, ec="none", zorder=3))
ax.add_patch(Rectangle((15.8, 19.9), 1.4, 0.45, fc="#e07b00", ec="none", zorder=4))
ax.text(16.5, 18.6, "1 expert label copied to every segment", ha="center", va="top", fontsize=5.8)
ax.text(1, 16.2, "→ one long recording, many label rows\n    (3–9 or ≥10 votes)", va="top", fontsize=6.2)
trace(ax, 12, 19, 9.5, seed=2)
span(ax, 12, 19, 7.2, "≈50 s")
ax.text(1, 4.6, "→ isolated excerpt: recording ≈ 50 s\n    (93% of 1–2-vote recordings)", va="top", fontsize=6.2)

# ---------- (b) a 200 s window shows where the file ends ----------
ax.text(34.5, 29.3, "(b) 200 s window: recording end visible", fontweight="bold", va="top", fontsize=6.8)
X0, X1 = 36, 64; C0, C1 = 46.5, 53.5           # 200 s window, central 50 s
ax.add_patch(Rectangle((C0, 19.8), C1 - C0, 4.4, fc=LAB, alpha=0.35, ec="none", zorder=0))
trace(ax, X0, X1, 22, seed=4); window(ax, X0, X1, 22)
ax.text(X1 + 0.5, 22, "real EEG\nthroughout", va="center", fontsize=5.8)
ax.add_patch(Rectangle((C0, 7.8), C1 - C0, 4.4, fc=LAB, alpha=0.35, ec="none", zorder=1))
zeros(ax, X0, C0, 10); zeros(ax, C1, X1, 10)
trace(ax, C0, C1, 10, seed=5); window(ax, X0, X1, 10)
ax.text(X0 + 5.2, 12.6, "zero-fill", ha="center", color=ZERO, fontsize=5.8)
ax.text(X1 - 5.2, 12.6, "zero-fill", ha="center", color=ZERO, fontsize=5.8)
span(ax, X0, X1, 7.4, "200 s window (central 50 s shaded)")
ax.text(X0, 4.6, "boundary visible → label mix of\nshort excerpts (Seizure 6.8% vs 15.4%\noverall; Other 53.3% vs 41.9%)", va="top", fontsize=6.2)

# ---------- (c) ±20 s shift does the same at 50 s; the fix ----------
ax.text(70, 29.3, "(c) So does a ±20 s shift at 50 s", fontweight="bold", va="top", fontsize=6.8)
F0, F1 = 74, 88; S = 5.6                        # 50 s file, 20 s shift (scale 50 s = 14 units)
d = trace(ax, F0, F1, 22, seed=6)
zeros(ax, F1, F1 + S, 22, h=1.6); window(ax, F0 + S, F1 + S, 22, h=1.9)
ax.text(F1 + S + 0.5, 22, "zeros\nat edge", va="center", fontsize=5.8, color=ZERO)
ax.text(F0 + S + 0.2, 24.6, "shifted window", fontsize=5.8)
trace(ax, F0, F1, 10, data=d)
mir = d[::-1][: int(len(d) * S / (F1 - F0))]
trace(ax, F1, F1 + S, 10, data=mir * (np.abs(d).max() / (np.abs(mir).max() + 1e-9)) * (np.abs(mir).max() / np.abs(d).max()), color=MIRROR)
window(ax, F0 + S, F1 + S, 10, h=1.9)
ax.text(F1 + S + 0.5, 10, "mirror\nfill", va="center", fontsize=5.8, color=MIRROR)
span(ax, F0, F1, 7.4, "50 s recording")
ax.text(70, 16.6, "zero padding → the cue reappears", fontsize=6.2, va="top")
ax.text(70, 4.6, "fix: mirror fill (+ random truncation\nfor long windows) hides\nthe boundary", va="top", fontsize=6.2)

for xs in (33, 68.5):
    ax.plot([xs, xs], [1, 29], color="0.8", lw=0.6)
fig.savefig("fig_mechanism_schematic.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("fig_mechanism_schematic.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
