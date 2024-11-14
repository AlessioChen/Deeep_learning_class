import matplotlib.pyplot as plt

import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

import seaborn as sns
sns.set_context("talk", font_scale=1.1)

NAMES = ["Disturbed Galaxies",
         "Merging Galaxies",
         "Round Smooth Galaxies",
         "In-between Round Smooth Galaxies",
         "Cigar Shaped Smooth Galaxies",
         "Barred Spiral Galaxies",
         "Unbarred Tight Spiral Galaxies",
         "Unbarred Loose Spiral Galaxies",
         "Edge-on Galaxies without Bulge",
         "Edge-on Galaxies with Bulge"]


def plot(imgs, labels, row_title=None, **imshow_kwargs):
    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    fig.set_figheight(12)
    fig.set_figwidth(24)

    for row_idx in range(len(imgs)):
        row = imgs[row_idx]
        for col_idx in range(len(row)):
            img = row[col_idx]
            # The label is one-hot...
            label = labels[row_idx][col_idx].argmax().item()
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set(xlabel=f"{label}:{NAMES[label]}")

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()

