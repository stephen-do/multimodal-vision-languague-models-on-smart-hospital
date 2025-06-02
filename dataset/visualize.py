import random
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

def coco_viz(
    img_dir: str | Path,
    ann_file: str | Path,
    class_id_map: dict | None = None,
    n_images: int = 5,
    show_masks: bool = True,
    alpha_mask: float = 0.4,
):
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    random.shuffle(img_ids)

    # Define some nice colors for bounding boxes
    color_list = [
        "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#46f0f0",
    ]

    def get_color(cat_id):
        return color_list[cat_id % len(color_list)]

    for img_id in img_ids[:n_images]:
        info = coco.loadImgs(img_id)[0]
        img_path = Path(img_dir) / info["file_name"]
        img = np.array(Image.open(img_path).convert("RGB"))

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        ax = plt.gca()

        for ann in anns:
            x, y, w, h = ann["bbox"]
            cat_id = ann["category_id"]
            color = get_color(cat_id)

            rect = Rectangle(
                (x, y),
                w,
                h,
                edgecolor=color,
                facecolor="none",
                linewidth=1.5,
                linestyle='-',
                alpha=0.9
            )
            ax.add_patch(rect)

            cat_name = (
                class_id_map.get(cat_id, coco.loadCats(cat_id)[0]["name"])
                if class_id_map
                else coco.loadCats(cat_id)[0]["name"]
            )

            ax.text(
                x,
                y - 5,
                cat_name,
                color="white",
                fontsize=10,
                fontweight="bold",
                bbox=dict(
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.7,
                    boxstyle="round,pad=0.3"
                ),
            )

            if show_masks and "segmentation" in ann and ann["segmentation"]:
                from pycocotools import mask as maskUtils

                rle = coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[:, :, 0] = 255  # red mask
                img_masked = np.where(m[..., None], colored_mask, 0)
                plt.imshow(img_masked, alpha=alpha_mask)

        plt.axis("off")
        plt.tight_layout()
        plt.show()

coco_viz(
    img_dir="vqc/test",
    ann_file="vqc/test/_annotations.coco.json",
    n_images=1,
    show_masks=False
)
