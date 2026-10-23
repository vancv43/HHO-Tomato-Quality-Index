"""Grad-CAM on the paper's trained ResNet18 (weights/cnn_best.pt); used by Fig.1 and Fig.9."""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

from common import CLASSES, WEIGHTS

IMG_SIZE = 128
device = torch.device("cpu")

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load(WEIGHTS, map_location=device))
    return m.to(device).eval()


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
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, F.softmax(out, dim=1)[0, class_idx].item()


def explain(cam_gen, path):
    """Return (resized RGB image, CAM, overlay, predicted index, confidence) for one image file."""
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    cam, pred_idx, conf = cam_gen.generate(tf(rgb).unsqueeze(0).to(device))
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = (0.45 * heatmap + 0.55 * rgb_resized).astype(np.uint8)
    return rgb_resized, cam, overlay, pred_idx, conf
