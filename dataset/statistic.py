# ========================================
# 1. IMPORT & SETUP
# ========================================
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pycocotools.coco import COCO
from tqdm import tqdm

# ========================================
# 2. OVERVIEW ANALYSIS
# ========================================
with open("vqc/train/_annotations.coco.json") as f:
    data = json.load(f)

print("Number of images:", len(data["images"]))
print("Number of annotations:", len(data["annotations"]))
print("Number of categories:", len(data["categories"]))

# ========================================
# 3. ANNOTATION PER IMAGE DISTRIBUTION
# ========================================
image_ids = [ann["image_id"] for ann in data["annotations"]]
counter = Counter(image_ids)

plt.figure()
plt.hist(counter.values(), bins=30)
plt.xlabel("Annotations per image")
plt.ylabel("Number of images")
plt.title("Distribution of annotations per image")
plt.tight_layout()
plt.show()

# ========================================
# 4. BOUNDING BOX SIZE DISTRIBUTION
# ========================================
widths = [ann["bbox"][2] for ann in data["annotations"]]
heights = [ann["bbox"][3] for ann in data["annotations"]]

plt.figure()
plt.scatter(widths, heights, alpha=0.2)
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Bounding Box Size Distribution")
plt.tight_layout()
plt.show()

# ========================================
# 5. CATEGORY DISTRIBUTION
# ========================================
category_ids = [ann["category_id"] for ann in data["annotations"]]
category_counts = Counter(category_ids)

category_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
category_names = [category_id_to_name[cid] for cid in category_counts]

plt.figure()
plt.bar(category_names, category_counts.values())
plt.xticks(rotation=45)
plt.ylabel("Number of annotations")
plt.title("Distribution by category")
plt.tight_layout()
plt.show()

# ========================================
# 6. COLLECT PIXEL INTENSITY BY CATEGORY
# ========================================
def collect_pixel_intensity_by_category(coco_json_path, image_dir):
    coco = COCO(coco_json_path)
    pixel_values = {"R": defaultdict(list), "G": defaultdict(list), "B": defaultdict(list)}
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    for ann_id in tqdm(coco.getAnnIds(), desc="Processing annotations"):
        ann = coco.loadAnns([ann_id])[0]
        cat_id = ann["category_id"]
        image_id = ann["image_id"]
        image_info = coco.loadImgs([image_id])[0]

        img_path = os.path.join(image_dir, image_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x, y, w, h = map(int, ann["bbox"])
        x1, y1 = max(0, x), max(0, y)
        x2 = min(x + w, rgb_img.shape[1])
        y2 = min(y + h, rgb_img.shape[0])
        crop = rgb_img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        R, G, B = cv2.split(crop)
        for channel, matrix in zip(["R", "G", "B"], [R, G, B]):
            norm = matrix.astype(np.float32) / 255.0
            pixel_values[channel][cat_id].extend(norm.flatten())

    return pixel_values, cat_id_to_name

def plot_pixel_histogram_by_category(pixel_values, cat_id_to_name, channel="R"):
    plt.figure(figsize=(12, 6))
    for cat_id, pixels in pixel_values[channel].items():
        label = cat_id_to_name.get(cat_id, f"ID {cat_id}")
        plt.hist(pixels, bins=50, alpha=0.5, label=label, density=True)
    plt.title(f"Normalized Pixel Intensity - Channel {channel}")
    plt.xlabel("Pixel Intensity (0-1)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

pixel_values, cat_id_to_name = collect_pixel_intensity_by_category(
    coco_json_path="vqc/test/_annotations.coco.json",
    image_dir="vqc/test"
)

plot_pixel_histogram_by_category(pixel_values, cat_id_to_name, channel="R")

# ========================================
# 7. CREATE BBOX HEATMAP
# ========================================
def create_bbox_heatmap(coco_json_path, image_dir, output_size=(512, 512), max_images=None):
    coco = COCO(coco_json_path)
    heatmap = np.zeros(output_size, dtype=np.float32)
    image_count = 0

    for img_info in tqdm(coco.dataset["images"], desc="Processing images"):
        img_path = os.path.join(image_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        h_org, w_org = img.shape[:2]
        scale_x = output_size[1] / w_org
        scale_y = output_size[0] / h_org

        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_info["id"]])):
            x, y, w, h = ann["bbox"]
            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + w) * scale_x)
            y2 = int((y + h) * scale_y)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(output_size[1], x2), min(output_size[0], y2)
            heatmap[y1:y2, x1:x2] += 1.0

        image_count += 1
        if max_images and image_count >= max_images:
            break

    heatmap /= heatmap.max()
    return heatmap

def plot_heatmap(heatmap, cmap='hot'):
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap=cmap)
    plt.colorbar(label='Normalized BBox Frequency')
    plt.title("BBox Heatmap")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

heatmap = create_bbox_heatmap(
    coco_json_path="vqc/test/_annotations.coco.json",
    image_dir="vqc/test",
    output_size=(512, 512),
    max_images=500
)

plot_heatmap(heatmap)
