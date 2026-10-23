"""Fig.2 dataset distribution, Fig.3 model comparison, Fig.4 HHO-ensemble confusion matrix.
Source: results/baseline_results.json (held-out validation split)."""
import numpy as np
import matplotlib.pyplot as plt

from common import CLASSES, load_json, save, clean_axes, confusion_plot

# ---- Fig.2: class distribution (train/val image counts of the dataset split) ----
train_counts = [949, 1992, 1975, 1585]
val_counts = [106, 222, 220, 177]
w = 0.35

fig, ax = plt.subplots(figsize=(6.2, 4.0))
x = np.arange(len(CLASSES))
b1 = ax.bar(x - w / 2, train_counts, w, label=f"Train (n={sum(train_counts)})", color="#2E7D8C")
b2 = ax.bar(x + w / 2, val_counts, w, label=f"Validation (n={sum(val_counts)})", color="#C97F0A")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 15, str(int(b.get_height())), ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_ylabel("Number of images")
ax.legend(frameon=False)
clean_axes(ax)
fig.tight_layout()
save(fig, "fig2_dataset_distribution.pdf")
plt.close(fig)

# ---- Fig.3: accuracy + macro-F1 per model ----
br = load_json("baseline_results.json")
names = list(br["per_model_metrics"].keys()) + ["HHO ensemble"]
accs = [br["per_model_metrics"][n]["accuracy"] for n in names[:-1]] + [br["ensemble_metrics"]["accuracy"]]
f1s = [br["per_model_metrics"][n]["f1_macro"] for n in names[:-1]] + [br["ensemble_metrics"]["f1_macro"]]

fig, ax = plt.subplots(figsize=(6.6, 4.0))
xx = np.arange(len(names))
b1 = ax.bar(xx - w / 2, accs, w, label="Accuracy", color="#1F4E5A")
b2 = ax.bar(xx + w / 2, f1s, w, label="Macro-F1", color="#8FBF6E")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006, f"{b.get_height()*100:.1f}", ha="center", fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
ax.set_ylim(0.8, 1.02)
ax.set_ylabel("Score")
ax.legend(frameon=False)
clean_axes(ax)
fig.tight_layout()
save(fig, "fig3_model_comparison.pdf")
plt.close(fig)


# ---- Fig.4: HHO ensemble confusion matrix ----
confusion_plot(br["ensemble_metrics"]["confusion_matrix"], CLASSES, "fig4_confusion_matrix_hho.pdf")
