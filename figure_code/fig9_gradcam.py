"""Fig.9 -- Grad-CAM of the paper's ResNet18, one validation image per class
(the images originally drawn with random.seed(7), now stored in sample_images/)."""
import matplotlib.pyplot as plt

from common import CLASSES, SAMPLE_IMAGES, save
from gradcam_utils import load_model, GradCAM, explain

model = load_model()
cam_gen = GradCAM(model, model.layer4[-1])

fig, axes = plt.subplots(len(CLASSES), 3, figsize=(7.2, 8.8))
for ci, cls in enumerate(CLASSES):
    rgb, cam, overlay, pred_idx, conf = explain(cam_gen, SAMPLE_IMAGES[cls])
    axes[ci, 0].imshow(rgb)
    axes[ci, 0].set_ylabel(cls, fontsize=11, fontweight="bold")
    axes[ci, 1].imshow(cam, cmap="jet")
    axes[ci, 2].imshow(overlay)
    for j in range(3):
        axes[ci, j].set_xticks([]); axes[ci, j].set_yticks([])
    pred = CLASSES[pred_idx]
    caption = f"pred: {pred} ({conf*100:.1f}%) {'✓' if pred == cls else '✗'}"
    if ci == 0:
        axes[ci, 0].set_title("Original", fontsize=10)
        axes[ci, 1].set_title("Grad-CAM", fontsize=10)
        axes[ci, 2].set_title(f"Overlay\n{caption}", fontsize=9)
    else:
        axes[ci, 2].set_title(caption, fontsize=9)

fig.tight_layout()
save(fig, "fig9_gradcam.pdf")
