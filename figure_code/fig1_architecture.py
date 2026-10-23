"""Fig.1 -- system architecture (Branch A: HHO-XQI, Branch B: ResNet18 + Grad-CAM).
The Grad-CAM thumbnail is computed live from the paper's trained ResNet18."""
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from common import SAMPLE_IMAGES, save
from gradcam_utils import load_model, GradCAM, explain

model = load_model()
_, _, real_overlay, _, _ = explain(GradCAM(model, model.layer4[-1]), SAMPLE_IMAGES["Ripe"])

TBOX, TEDGE = "#E3F2F0", "#1F6B66"
OBOX, OEDGE = "#FDF0E0", "#C97F0A"
PBOX, PEDGE = "#F3EAF7", "#8B5CA6"
GBOX, GEDGE = "#E7F5EA", "#2E7D46"
CBOX, CEDGE = "#E7EEF9", "#3A5A9B"
BANDA = "#F4FBFA"
BANDB = "#F2F5FB"

YB = 2.6          # vertical centre of Branch B
Y0, Y1 = 1.45, 8.05
fig, ax = plt.subplots(figsize=(12.4, Y1 - Y0))
ax.set_xlim(0, 12.4)
ax.set_ylim(Y0, Y1)
ax.axis("off")
ax.set_aspect("equal")

ax.add_patch(Rectangle((0.15, 3.9), 12.1, 4.0, facecolor=BANDA, edgecolor="none", zorder=0))  # Branch A band
ax.add_patch(Rectangle((0.15, YB - 1.0), 12.1, 2.0, facecolor=BANDB, edgecolor="none", zorder=0))  # Branch B band


def box(cx, cy, w, h, text, fc, ec, fontsize=9.2, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.09",
                                linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            zorder=4, linespacing=1.35, fontweight=weight)
    return cx, cy, w, h


def arrow(p1, p2, color="#333333", lw=1.5, style="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                                 linestyle=style, color=color, shrinkA=2, shrinkB=2, zorder=2))


def photo_grid(cx, cy, cell=0.58, gap=0.06):
    offsets = {"Damaged": (-1, 1), "Old": (1, 1), "Ripe": (-1, -1), "Unripe": (1, -1)}
    for cls, (ox, oy) in offsets.items():
        img_rgb = cv2.cvtColor(cv2.imread(SAMPLE_IMAGES[cls]), cv2.COLOR_BGR2RGB)
        ab = AnnotationBbox(OffsetImage(img_rgb, zoom=0.185),
                            (cx + ox * (cell / 2 + gap / 2), cy + oy * (cell / 2 + gap / 2)),
                            frameon=True, pad=0.02, bboxprops=dict(edgecolor=TEDGE, linewidth=1.1))
        ax.add_artist(ab)
    ax.add_patch(Rectangle((cx - cell - gap * 1.2, cy - cell - gap * 1.2),
                           2 * cell + 2.4 * gap, 2 * cell + 2.4 * gap,
                           fill=False, edgecolor=TEDGE, linewidth=1.6, zorder=5))


# ---- shared input ----
photo_grid(1.35, 6.9)
b_pre = box(4.15, 6.9, 1.95, 0.85, "Preprocessing\nresize 128$\\times$128,\ncolor-space convert", TBOX, TEDGE, 8.3)
arrow((1.35 + 0.7, 6.9), (b_pre[0] - b_pre[2] / 2, 6.9))

# ---- Branch A ----
b_feat = box(4.15, 5.60, 2.15, 1.0, "125-d color-statistical\nfeatures (RGB+HSV+Lab,\nhistograms, ripeness idx.)", TBOX, TEDGE, 8.3)
arrow((b_pre[0], 6.9 - 0.425), (4.15, 5.60 + 0.5))

b_lr = box(6.75, 6.70, 1.75, 0.62, "Logistic\nRegression", TBOX, TEDGE, 8.2)
b_rf = box(6.75, 5.97, 1.75, 0.62, "Random\nForest", TBOX, TEDGE, 8.2)
b_et = box(6.75, 5.24, 1.75, 0.62, "Extra\nTrees", TBOX, TEDGE, 8.2)
b_knn = box(6.75, 4.51, 1.75, 0.62, "K-Nearest\nNeighbors", TBOX, TEDGE, 8.2)
for b in (b_lr, b_rf, b_et, b_knn):
    arrow((b_feat[0] + b_feat[2] / 2, 5.60), (b[0] - b[2] / 2, b[1]))

b_hho = box(9.45, 5.60, 1.9, 1.35, "HHO ensemble-\nweight optimizer\n(Eq. 1-3)", OBOX, OEDGE, 8.6)
for b in (b_lr, b_rf, b_et, b_knn):
    arrow((b[0] + b[2] / 2, b[1]), (b_hho[0] - b_hho[2] / 2, b_hho[1]))

b_xqi = box(11.25, 5.60, 1.25, 1.8, "Tomato\nQuality\nIndex +\ngrade\n(Eq. 4)", GBOX, GEDGE, 8.2)
arrow((b_hho[0] + b_hho[2] / 2, 5.60), (b_xqi[0] - b_xqi[2] / 2, 5.60))

# same images also feed Branch B (dashed connector down the left lane)
arrow((1.35, 6.9 - 0.71), (1.35, YB + 0.425), style=(0, (3, 2)))

# ---- Branch B ----
b_cnn = box(1.85, YB, 1.95, 0.85, "ResNet18 CNN\n(fine-tuned,\nImageNet init.)", CBOX, CEDGE, 8.6)
b_grad = box(4.45, YB, 1.85, 0.85, "Grad-CAM\nvisual\nexplanation", PBOX, PEDGE, 8.6)
arrow((b_cnn[0] + b_cnn[2] / 2, YB), (b_grad[0] - b_grad[2] / 2, YB))

ax.add_artist(AnnotationBbox(OffsetImage(real_overlay, zoom=0.42), (6.85, YB), frameon=True, pad=0.03,
                             bboxprops=dict(edgecolor=PEDGE, linewidth=1.3)))
arrow((b_grad[0] + b_grad[2] / 2, YB), (6.36, YB))

b_out = box(9.55, YB, 1.55, 0.9, "4-class grade\n+ heatmap\noverlay", GBOX, GEDGE, 8.2)
arrow((7.34, YB), (b_out[0] - b_out[2] / 2, YB))

fig.tight_layout()
save(fig, "fig1_architecture.pdf")
