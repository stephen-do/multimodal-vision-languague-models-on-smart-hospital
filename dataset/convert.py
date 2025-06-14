import json
import random
from collections import defaultdict
from transformers import RobertaTokenizerFast

PHRASES = {
    1: ["mild periapical inflammation", "mild PAI", "a slightly inflamed tooth root"],
    2: ["moderate periapical inflammation", "moderate PAI", "a moderately inflamed periapex"],
    3: ["severe periapical inflammation", "severe PAI", "an extensively inflamed root apex"]
}

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def convert(input_path, output_path):
    with open(input_path, "r") as f:
        coco = json.load(f)

    img_id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_id_to_anns[ann["image_id"]].append(ann)

    for img in coco["images"]:
        image_id = img["id"]
        anns = img_id_to_anns[image_id]

        phrase_assignments = {}
        caption_phrases = []

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in PHRASES:
                continue
            phrase = random.choice(PHRASES[cat_id])
            phrase_assignments[ann["id"]] = phrase
            caption_phrases.append(phrase)

        caption = ". ".join(caption_phrases) + "." if caption_phrases else "No visible inflammation."
        img["caption"] = caption
        img["dataset_name"] = "periapical_inflammation"

        encoding = tokenizer(caption, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding["offset_mapping"]

        for ann in anns:
            ann_id = ann["id"]
            if ann_id not in phrase_assignments:
                ann["tokens_positive"] = []
                continue

            phrase = phrase_assignments[ann_id]
            start = caption.find(phrase)
            end = start + len(phrase)

            indices = [i for i, (s, e) in enumerate(offsets) if s >= start and e <= end]

            # Group consecutive tokens into (start, end) ranges
            ranges = []
            if indices:
                b = prev = indices[0]
                for i in indices[1:]:
                    if i == prev + 1:
                        prev = i
                    else:
                        ranges.append((b, prev + 1))
                        b = prev = i
                ranges.append((b, prev + 1))

            ann["tokens_positive"] = ranges

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ Done! Converted file saved to: {output_path}")

# Example usage:
for i in ['train', 'test', 'valid']:
    convert(f"vqc/{i}/_annotations.coco.json", f"vqc/{i}/_annotations.json")
