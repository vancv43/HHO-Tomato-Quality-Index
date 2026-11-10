# /*
#  * Van Van Chau(1), Phayung Meesad(2,*), and Minh Tuan Nguyen(1)
#  * (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#  *     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
#  * (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#  *     Bangkok, Thailand, phayung.m@itd.kmutnb.ac.th
#  * (*) Corresponding author
#  */
import os
import time
import json
import torch
import torch.nn as nn
from torchvision import models

OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
CLASSES = ["Damaged", "Old", "Ripe", "Unripe"]


def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


if __name__ == "__main__":
    # CPU latency (fair comparison with the classical pipeline, also measured on CPU)
    device_cpu = torch.device("cpu")
    model = build_model().to(device_cpu)
    model.load_state_dict(torch.load(f"{OUT}/cnn_best.pt", map_location=device_cpu))
    model.eval()

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        for _ in range(5):
            model(x)  # warmup
        n = 50
        t0 = time.time()
        for _ in range(n):
            model(x)
        cpu_ms = (time.time() - t0) / n * 1000
    print(f"CNN CPU inference: {cpu_ms:.3f} ms/image (batch=1, single thread default)")

    # GPU latency, if a GPU is available
    gpu_ms = None
    gpu = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
    if gpu:
        synchronize = torch.cuda.synchronize if gpu == "cuda" else torch.mps.synchronize
        device_gpu = torch.device(gpu)
        model_gpu = build_model().to(device_gpu)
        model_gpu.load_state_dict(torch.load(f"{OUT}/cnn_best.pt", map_location=device_gpu))
        model_gpu.eval()
        xg = torch.randn(1, 3, 128, 128).to(device_gpu)
        with torch.no_grad():
            for _ in range(10):
                model_gpu(xg)
            synchronize()
            n = 100
            t0 = time.time()
            for _ in range(n):
                model_gpu(xg)
            synchronize()
            gpu_ms = (time.time() - t0) / n * 1000
        print(f"CNN GPU inference: {gpu_ms:.3f} ms/image (batch=1)")

    with open(f"{OUT}/cnn_latency.json", "w") as f:
        json.dump({"cpu_ms_per_image": cpu_ms, "gpu_ms_per_image": gpu_ms}, f, indent=2)
    print("Saved cnn_latency.json")
