import json
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os

# CATEGORY MAP: Chuyển category_id thành phrase
CATEGORY_MAP = {
    1: "mild pai",
    2: "moderate pai",
    3: "severe pai"
}

# Loop qua từng tập dữ liệu
for set_ in ['train', 'valid', 'test']:
    with open(f'vqc/{set_}/_annotations.coco.json', 'r') as f:
        coco = json.load(f)

    output = {
        "images": [],
        "annotations": [],
        "sentences": [],
        "categories": [
            {"id": 1, "name": "mild pai"},
            {"id": 2, "name": "moderate pai"},
            {"id": 3, "name": "severe pai"},
        ],
        "info": {
            "year": "2025",
            "version": "1",
            "description": "Converted to MDETR RefExp format",
            "contributor": "You",
            "url": "https://your.dataset.source/",
            "date_created": "2025-06-24"
        }
    }

    ann_id = 0
    for img in tqdm(coco['images'], desc=f"Processing {set_} set"):
        image_id = img['id']

        anns = [a for a in coco['annotations'] if a['image_id'] == image_id]
        if not anns:
            continue

        phrases = [CATEGORY_MAP[a['category_id']] for a in anns]
        caption = ", ".join(phrases)

        output['images'].append({
            "id": image_id,
            "file_name": img['file_name'],
            "height": img['height'],
            "width": img['width'],
            "caption": caption,
            "dataset_name": 'vqc'
        })

        tokens = word_tokenize(caption)

        # Mapping phrase -> token spans (list of token indices)
        token_map = []
        used_spans = set()
        for phrase in phrases:
            phrase_tokens = word_tokenize(phrase)
            matched = False
            for i in range(len(tokens) - len(phrase_tokens) + 1):
                token_indices = list(range(i, i + len(phrase_tokens)))
                if tokens[i:i + len(phrase_tokens)] == phrase_tokens and tuple(token_indices) not in used_spans:
                    token_map.append(token_indices)  # ✅ đúng format RefExp
                    used_spans.add(tuple(token_indices))
                    matched = True
                    break
            if not matched:
                print(f"[WARN] Phrase not found in caption: '{phrase}' → skipped.")
                token_map.append(None)

        for a, token_span in zip(anns, token_map):
            if token_span is None:
                continue  # Skip nếu không match được phrase trong caption
            x, y, w, h = a['bbox']
            bbox_xyxy = [x, y, x + w, y + h]
            output['annotations'].append({
                "id": ann_id,
                "image_id": image_id,
                "bbox": bbox_xyxy,
                "category_id": a['category_id'],
                "iscrowd": a.get("iscrowd", 0),
                "area": a.get("area", a['bbox'][2] * a['bbox'][3]),
                "tokens_positive": [token_span],  # ✅ đúng format MDETR
                "original_caption": caption,
                "dataset_name": 'vqc'
            })
            ann_id += 1

        output['sentences'].append({
            "image_id": image_id,
            "caption": caption,
            "tokens": tokens
        })

    # Ghi file JSON ra đúng format
    out_file = f'vqc/{set_}/mdetr_periapical_annotations.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[✓] Saved: {out_file}")
