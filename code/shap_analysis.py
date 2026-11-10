# Van Van Chau(1), Phayung Meesad(2,*), and Minh Tuan Nguyen(1)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Bangkok, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
import os
import json
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

from features import FEATURE_NAMES

OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
SEED = 42
CLASSES = ["Damaged", "Old", "Ripe", "Unripe"]

if __name__ == "__main__":
    data = np.load(f"{OUT}/features.npz")
    Xtr, ytr, Xval, yval = data["Xtr"], data["ytr"], data["Xval"], data["yval"]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xval_s = scaler.transform(Xtr), scaler.transform(Xval)

    clf = ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    clf.fit(Xtr_s, ytr)

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(Xval_s), size=min(300, len(Xval_s)), replace=False)
    Xsample = Xval_s[sample_idx]

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xsample)
    # sv shape can be (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
    sv = np.array(sv)
    if sv.shape[0] == len(CLASSES):
        sv = np.transpose(sv, (1, 2, 0))  # -> (n_samples, n_features, n_classes)
    print("shap values shape (samples, features, classes):", sv.shape)

    mean_abs_per_class = np.mean(np.abs(sv), axis=0)  # (features, classes)
    mean_abs_global = mean_abs_per_class.mean(axis=1)  # (features,)

    order = np.argsort(-mean_abs_global)
    top_n = 15
    top_idx = order[:top_n]

    print("\nTop features (global mean |SHAP|):")
    for i in top_idx:
        print(f"  {FEATURE_NAMES[i]:20s} {mean_abs_global[i]:.5f}")

    # per-class top-5 table
    per_class_top = {}
    for ci, cname in enumerate(CLASSES):
        order_c = np.argsort(-mean_abs_per_class[:, ci])[:5]
        per_class_top[cname] = [(FEATURE_NAMES[i], float(mean_abs_per_class[i, ci])) for i in order_c]
        print(f"\nTop-5 for class {cname}:")
        for name, val in per_class_top[cname]:
            print(f"  {name:20s} {val:.5f}")

    out = {
        "global_top15": [(FEATURE_NAMES[i], float(mean_abs_global[i])) for i in top_idx],
        "per_class_top5": per_class_top,
    }
    with open(f"{OUT}/shap_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved shap_results.json and fig_shap_importance.png")
