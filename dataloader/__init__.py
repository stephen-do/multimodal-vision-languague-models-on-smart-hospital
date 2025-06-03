import torch.utils.data
import torchvision
from .coco import build as build_coco
from .refexp import build as build_refexp


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, (torchvision.datasets.CocoDetection, )):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == "coco":
        return build_coco(image_set, args)
    if dataset_file == "refexp":
        return build_refexp(image_set, args)
    raise ValueError(f"dataset {dataset_file} not supported")