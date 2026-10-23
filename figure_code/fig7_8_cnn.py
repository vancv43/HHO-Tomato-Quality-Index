"""Fig.7 ResNet18 training/validation curves and Fig.8 ResNet18 confusion matrix.
Source: results/cnn_history.json."""
import numpy as np
import matplotlib.pyplot as plt

from common import load_json, save, clean_axes, confusion_plot

h = load_json("cnn_history.json")
hist = h["history"]
epochs = np.arange(1, len(hist["train_acc"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
for ax, key, ylabel, title in [(axes[0], "acc", "Accuracy", "Accuracy"),
                               (axes[1], "loss", "Cross-entropy loss", "Loss")]:
    ax.plot(epochs, hist[f"train_{key}"], color="#1F4E5A", lw=2, label="Train")
    ax.plot(epochs, hist[f"val_{key}"], color="#2E7D8C", lw=2, ls="--", label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color="#1F4E5A")
    ax.legend(frameon=False, fontsize=8.5)
    clean_axes(ax)
fig.tight_layout()
save(fig, "fig7_cnn_curves.pdf", bbox_inches="tight")
plt.close(fig)

confusion_plot(h["confusion_matrix"], h["classes_imagefolder_order"], "fig8_cnn_confusion_matrix.pdf")
