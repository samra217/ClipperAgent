import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
FEATS = BASE_DIR / "data_insamling" / "features"
HEATMAPS = BASE_DIR / "data_insamling" / "heatmaps"  
PLOTS = BASE_DIR / "data_insamling" / "plots"

INK = "#2b2b2b"
MUTED = "#8a8a8a"
LINE = "#1f5fa8"
FLOOR = 0.3  


def fmt_time(x, _=None):
    x = int(x)
    return f"{x // 60}:{x % 60:02d}"


def main():
    vid = sys.argv[1] if len(sys.argv) > 1 else sorted(FEATS.glob("*_sounds.npz"))[0].stem.replace("_sounds", "")

    sounds = np.load(FEATS / f"{vid}_sounds.npz")
    X, cols, duration = sounds["X"], [str(c) for c in sounds["cols"]], float(sounds["duration"])
    n_bins = X.shape[1]
    groups = [(c, X[i]) for i, c in enumerate(cols) if c.endswith("_max")]

    text_path = FEATS / f"{vid}_text.npz"
    text = np.load(text_path) if text_path.exists() else None

    hm_path = HEATMAPS / f"{vid}.json"
    heatmap = json.loads(hm_path.read_text()) if hm_path.exists() else None

    panels = ["sounds"] + (["text"] if text is not None else []) + (["heatmap"] if heatmap else [])
    ratios = {"sounds": 3.2, "text": 2.0, "heatmap": 1.4}
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 2 + sum(ratios[p] for p in panels)),
                             sharex=True, gridspec_kw={"height_ratios": [ratios[p] for p in panels]})
    axes = np.atleast_1d(axes)
    extent = [0, duration, 0, 1]

    for ax, kind in zip(axes, panels):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=9)

        if kind == "sounds":
            
            M = np.stack([v / max(v.max(), FLOOR) for _, v in groups])
            ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
                      extent=[0, duration, len(groups), 0])
            ax.set_yticks(np.arange(len(groups)) + 0.5)
            ax.set_yticklabels([f"{c[:-4]}  (max {v.max():.2f})" for c, v in groups], fontsize=9, color=INK)
            ax.set_title(f"{vid}  ·  {fmt_time(duration)}  ·  {n_bins} bins à {duration / n_bins:.2f} s",
                         loc="left", fontsize=11, color=INK)
            ax.text(1.0, 1.02, "ljudgrupper (PANNs), mörkare = högre sannolikhet", transform=ax.transAxes,
                    ha="right", fontsize=8, color=MUTED)

        elif kind == "text":
            t = np.linspace(0, duration, n_bins, endpoint=False) + duration / n_bins / 2
            tc = [str(c) for c in text["cols"]]
            Xt = text["X"]
            for name, color, ls in [("words_per_s", LINE, "-"), ("hype_words", "#c0392b", "-")]:
                if name in tc:
                    v = Xt[tc.index(name)]
                    ax.plot(t, v / (v.max() + 1e-9), color=color, lw=1.4, ls=ls, label=name)
            if "speech-frac" in tc:
                ax.fill_between(t, Xt[tc.index("speech-frac")], color=MUTED, alpha=0.25, lw=0, label="speech-frac")
            if "excl_q_count" in tc:
                v = Xt[tc.index("excl_q_count")]
                ax.plot(t, v / (v.max() + 1e-9), color="#e08a1e", lw=1.4, label="excl_q_count")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("skalad 0–1", fontsize=9, color=INK)
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=8, frameon=False)

        else:  # heatmap
            t = [(h["start_time"] + h["end_time"]) / 2 for h in heatmap]
            ax.fill_between(t, [h["value"] for h in heatmap], step="mid", color=LINE, alpha=0.8, lw=0)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("replay (YouTube)", fontsize=9, color=INK)

    axes[-1].set_xlim(0, duration)
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(fmt_time))
    axes[-1].set_xlabel("tid i videon", fontsize=9, color=INK)
    fig.tight_layout()

    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / f"{vid}.png"
    fig.savefig(out, dpi=150)
    print(f"Sparade {out}")
    plt.show()


if __name__ == "__main__":
    main()