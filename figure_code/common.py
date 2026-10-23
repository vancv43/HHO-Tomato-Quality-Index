"""Shared paths and plotting settings for the manuscript figures.

All numbers come from the archived experiment outputs in ../results and the
trained CNN in ../weights.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))     # archived results
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))     # output used by the .tex
SAMPLES = os.path.join(HERE, "sample_images")                       # 4 validation images (Fig.1, Fig.9)
WEIGHTS = os.path.normpath(os.path.join(HERE, "..", "weights", "cnn_best.pt"))

DPI = 300
CLASSES = ["Damaged", "Old", "Ripe", "Unripe"]
SAMPLE_IMAGES = {c: os.path.join(SAMPLES, f) for c, f in
                 {"Damaged": "Damaged.png", "Old": "Old.jpg", "Ripe": "Ripe.jpg", "Unripe": "Unripe.png"}.items()}


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def save(fig, name, **kw):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, **kw)
    print("Saved", path)


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def confusion_plot(cm, labels, name):
    """Row-normalised confusion matrix with counts and row % (Fig.4, Fig.8)."""
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            pct = cm[i, j] / cm[i].sum() * 100
            ax.text(j, i, f"{cm[i, j]}\n({pct:.0f}%)", ha="center", va="center", fontsize=9,
                    color="white" if pct > 50 else "#222222")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    save(fig, name)
    plt.close(fig)
