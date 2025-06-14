# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import copy
from collections import defaultdict
from pathlib import Path
import torch
import torch.utils.data
from transformers import RobertaTokenizerFast
from .coco import ModulatedDetection, make_coco_transforms


class PhraseGroundingPAIDetection(ModulatedDetection):
    pass

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    pg_pai_dataset_name = args.pg_pai_dataset_name
    if pg_pai_dataset_name not in ["pg_pai"]:
        assert False, f"{pg_pai_dataset_name} not a valid datasset name for refexp"
    PATHS = {
        "train": (root / "train", root / "train" / f"_annotations.json"),
        "val": (root / "valid", root / "valid" / f"_annotations.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = PhraseGroundingPAIDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=False,
        return_tokens=True,
        tokenizer=tokenizer,
    )
    return dataset
