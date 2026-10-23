"""Fig.5 HHO convergence and Fig.6 SHAP top-15 features.
Plotted from the ARCHIVED results (results/convergence_results.json, results/shap_results.json) that the
manuscript text is based on. To recompute them from scratch run code/convergence.py and
code/shap_analysis.py (exact with scikit-learn 1.5.2, see requirements.txt)."""
import numpy as np
import matplotlib.pyplot as plt

from common import load_json, save, clean_axes

# ---- Fig.5 ----
curves = np.array(load_json("convergence_results.json")["curves"])  # (10 seeds, 41) best (1 - macro-F1)
mean_curve, std_curve = curves.mean(axis=0), curves.std(axis=0)
iters = np.arange(len(mean_curve))

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(iters, 1 - mean_curve, color="#1F4E5A", lw=2, label="Mean best macro-F1 (10 seeds)")
ax.fill_between(iters, 1 - (mean_curve - std_curve), 1 - (mean_curve + std_curve),
                color="#1F4E5A", alpha=0.18, label="$\\pm$1 SD")
ax.set_xlabel("HHO iteration")
ax.set_ylabel("Best validation macro-F1 found so far")
ax.legend(fontsize=8.5, frameon=False)
clean_axes(ax)
fig.tight_layout()
save(fig, "fig5_hho_convergence.pdf")
plt.close(fig)

# ---- Fig.6 ----
top = load_json("shap_results.json")["global_top15"]
fig, ax = plt.subplots(figsize=(6.5, 5.0))
ax.barh([n for n, _ in top][::-1], [v for _, v in top][::-1], color="#2E7D8C")
ax.set_xlabel("Mean |SHAP value| (avg. over 4 classes)")
fig.tight_layout()
save(fig, "fig6_shap_importance.pdf")
plt.close(fig)
