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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from hho import hho_minimize
from train_baseline import make_models, softmax_simplex

OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
SEED = 42
MAX_ITER = 40
N_SEEDS = 10

if __name__ == "__main__":
    data = np.load(f"{OUT}/features.npz")
    Xtr_full, ytr_full, Xval, yval = data["Xtr"], data["ytr"], data["Xval"], data["yval"]
    Xtr, Xhho, ytr, yhho = train_test_split(Xtr_full, ytr_full, test_size=0.15,
                                             stratify=ytr_full, random_state=SEED)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xhho_s = scaler.transform(Xtr), scaler.transform(Xhho)

    models = make_models(SEED)
    proba_hho = {}
    for name, clf in models.items():
        clf.fit(Xtr_s, ytr)
        proba_hho[name] = clf.predict_proba(Xhho_s)
    model_names = list(models.keys())

    def fitness(w):
        w = softmax_simplex(w)
        combined = sum(w[i] * proba_hho[n] for i, n in enumerate(model_names))
        pred = combined.argmax(axis=1)
        return 1.0 - f1_score(yhho, pred, average="macro", zero_division=0)

    curves = []
    for seed in range(N_SEEDS):
        _, _, curve = hho_minimize(fitness, dim=len(model_names), lb=0.0, ub=1.0,
                                    n_hawks=16, max_iter=MAX_ITER, seed=seed, log_convergence=True)
        curves.append(curve)
        print(f"seed {seed}: final fitness={curve[-1]:.5f} (1-F1)")

    curves = np.array(curves)  # (N_SEEDS, MAX_ITER+1)
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)
    mean_f1 = 1 - mean_curve

    with open(f"{OUT}/convergence_results.json", "w") as f:
        json.dump({"curves": curves.tolist(), "mean_f1_per_iter": mean_f1.tolist()}, f, indent=2)
    print("Saved convergence_results.json")
