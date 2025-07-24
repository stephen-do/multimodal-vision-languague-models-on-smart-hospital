import json
from tqdm import tqdm
import os

# CATEGORY MAP
CATEGORY_MAP = {
    1: "pai 3",
    2: "pai 4",
    3: "pai 5"
}

for set_ in ['train', 'valid', 'test']:
    with open(f'vqc/{set_}/_annotations.coco.json', 'r') as f:
        coco = json.load(f)

    output = {
        "images": [],
        "annotations": [],
        "sentences": [],
        "categories": [
            {"id": 1, "name": "pai 3"},
            {"id": 2, "name": "pai 4"},
            {"id": 3, "name": "pai 5"},
        ],
        "info": {
            "year": "2025",
            "version": "1",
            "description": "Converted to MDETR RefExp format with unique caption per image",
            "contributor": "You",
            "url": "https://your.dataset.source/",
            "date_created": "2025-07-10"
        }
    }

    ann_id = 0
    for img in tqdm(coco['images'], desc=f"Processing {set_} set"):
        image_id = img['id']
        anns = [a for a in coco['annotations'] if a['image_id'] == image_id]
        if not anns:
            continue

        # Duyệt theo thứ tự xuất hiện, lọc trùng
        seen = set()
        unique_phrases = []
        for a in anns:
            phrase = CATEGORY_MAP.get(a['category_id'], None)
            if phrase and phrase not in seen:
                unique_phrases.append(phrase)
                seen.add(phrase)

        # Caption dạng "pai 3, pai 4, pai 5"
        caption = ", ".join(unique_phrases)

        # Tính span chính xác không bao gồm dấu phẩy
        phrase_to_span = {}
        start = 0
        for phrase in unique_phrases:
            search_phrase = phrase.strip()
            idx = caption.find(search_phrase, start)
            if idx == -1:
                print(f"[WARN] Phrase '{phrase}' not found in caption: '{caption}'")
                continue
            phrase_to_span[phrase] = [(idx, idx + len(search_phrase) - 1)]
            start = idx + len(search_phrase) - 1

        # Ghi ảnh
        output['images'].append({
            "id": image_id,
            "file_name": img['file_name'],
            "height": img['height'],
            "width": img['width'],
            "caption": caption,
            "dataset_name": 'vqc'
        })

        # Ghi annotation
        for a in anns:
            category_id = a['category_id']
            phrase = CATEGORY_MAP.get(category_id, None)
            if phrase not in phrase_to_span:
                continue

            # x, y, w, h = a['bbox']
            # bbox_xyxy = [x, y, x + w, y + h]

            output['annotations'].append({
                "id": ann_id,
                "image_id": image_id,
                "bbox": a['bbox'],
                "category_id": category_id,
                "iscrowd": a.get("iscrowd", 0),
                "area": a.get("area"),
                "tokens_positive": phrase_to_span[phrase],
                "original_caption": caption,
                "dataset_name": 'vqc'
            })
            ann_id += 1

        # Ghi sentence
        output['sentences'].append({
            "image_id": image_id,
            "caption": caption
        })

    # Lưu file
    out_file = f'vqc/{set_}/mdetr_periapical_annotations.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[✓] Saved: {out_file}")
