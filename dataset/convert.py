import json
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk

# nltk.download('punkt_tab')
# Map category_id to disease severity
for set_ in ['train', 'test', 'valid']:
    CATEGORY_MAP = {
        1: "mild periapical inflammation",
        2: "moderate periapical inflammation",
        3: "severe periapical inflammation"
    }

    with open(f'vqc/{set_}/_annotations.coco.json', 'r') as f:
        coco = json.load(f)

    output = {
        "images": [],
        "annotations": [],
        "sentences": []
    }

    ann_id = 0
    for img in tqdm(coco['images']):
        image_id = img['id']

        anns = [a for a in coco['annotations'] if a['image_id'] == image_id]
        phrases = [CATEGORY_MAP[a['category_id']] for a in anns]
        caption = ", ".join(phrases)
        img["caption"] = caption
        output['images'].append({
            "id": image_id,
            "file_name": img['file_name'],
            "height": img['height'],
            "width": img['width'],
            "caption": img['caption'],
            "dataset_name": 'vqc',
        })
        tokens = word_tokenize(caption)

        # Handle repeated phrases (e.g., 2 x mild)
        token_map = []
        used_spans = set()
        for phrase in phrases:
            phrase_tokens = word_tokenize(phrase)
            for i in range(len(tokens) - len(phrase_tokens) + 1):
                if tokens[i:i + len(phrase_tokens)] == phrase_tokens and (i, i + len(phrase_tokens)) not in used_spans:
                    token_map.append([i, i + len(phrase_tokens)])
                    used_spans.add((i, i + len(phrase_tokens)))
                    break
            else:
                token_map.append(None)

        for a, token_span in zip(anns, token_map):
            if token_span is None:
                continue
            output['annotations'].append({
                "image_id": image_id,
                "bbox": a['bbox'],
                "category_id": a['category_id'],
                "id": ann_id,
                "iscrowd": a.get("iscrowd", 0),
                "area": a.get("area", a['bbox'][2] * a['bbox'][3]),
                "tokens_positive": [token_span],
                "original_caption": caption,
                "dataset_name": 'vqc',
            })
            ann_id += 1

        output['sentences'].append({
            "image_id": image_id,
            "caption": caption,
            "tokens": tokens
        })
        output['categories'] = [
            {
                "id": 1,
                "name": "3"
            },
            {
                "id": 2,
                "name": "4"
            },
            {
                "id": 3,
                "name": "5"
            }
        ]

    # Save JSON
    with open(f'vqc/{set_}/mdetr_periapical_annotations.json', 'w') as f:
        json.dump(output, f, indent=2)
