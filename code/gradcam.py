# /*
#  * Van Van Chau(1), Phayung Meesad(2,*), and Minh Tuan Nguyen(1)
#  * (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#  *     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
#  * (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#  *     Bangkok, Thailand, phayung.m@itd.kmutnb.ac.th
#  * (*) Corresponding author
#  */
import os
import glob
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms, models

DATA_ROOT = os.environ.get("TOMATO_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_raw", "Tomatoes Dataset"))
OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
CLASSES = ["Damaged", "Old", "Ripe", "Unripe"]  # ImageFolder alphabetical order, matches train_cnn.py
IMG_SIZE = 128
SEED = 7

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=None):
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        score = out[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, F.softmax(out, dim=1)[0, class_idx].item()


if __name__ == "__main__":
    model = build_model().to(device)
    model.load_state_dict(torch.load(f"{OUT}/cnn_best.pt", map_location=device))
    model.eval()
    cam_gen = GradCAM(model, model.layer4[-1])

    random.seed(SEED)
    cams, results = {}, {}
    for ci, cls in enumerate(CLASSES):
        files = sorted(glob.glob(f"{DATA_ROOT}/val/{cls}/*"))
        f = random.choice(files)
        bgr = cv2.imread(f)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        x = tf(rgb).unsqueeze(0).to(device)
        x.requires_grad_(False)
        cam, pred_idx, conf = cam_gen.generate(x)
        cams[cls] = cam
        results[cls] = {"image": os.path.basename(f), "predicted": CLASSES[pred_idx], "confidence": conf}
        print(f"{cls}: predicted {CLASSES[pred_idx]} ({conf*100:.1f}%)")

    np.savez_compressed(f"{OUT}/gradcam_maps.npz", **cams)
    with open(f"{OUT}/gradcam_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("Saved gradcam_maps.npz and gradcam_results.json")
