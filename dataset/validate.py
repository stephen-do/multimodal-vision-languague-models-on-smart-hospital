import argparse
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from dataloader.refexp import build


def show_image_with_boxes(image, target):
    image = F.to_pil_image(image)
    plt.imshow(image)

    for box in target["boxes"]:
        x0, y0, x1, y1 = box.tolist()
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="red", linewidth=2)
        plt.gca().add_patch(rect)

    if "caption" in target:
        plt.title(target["caption"])

    plt.axis("off")
    plt.show()


def main(args):
    class FakeArgs:
        def __init__(self, coco_path, refexp_ann_path):
            self.coco_path = coco_path
            self.refexp_ann_path = refexp_ann_path
            self.refexp_dataset_name = "vqc"
            self.test = False
            self.test_type = "test"
            self.masks = False
            self.text_encoder_type = "roberta-base"

    fake_args = FakeArgs(args.coco_path, args.ann_path)

    dataset = build("train", fake_args)

    print(f"Dataset loaded: {len(dataset)} samples")

    sample = dataset[0]

    image = sample[0]
    target = sample[1]
    print("\n--- Sample Info ---")
    print("Image shape:", image.shape)
    print("Target keys:", target.keys())
    print("Caption:", target.get("caption", "N/A"))
    print("Boxes:", target["boxes"])

    if "input_ids" in target:
        print("Tokenized input_ids shape:", target["input_ids"].shape)
    else:
        print("input_ids not found in target!")

    show_image_with_boxes(image, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_path", default='dataset/vqc/train',type=str, required=False, help="Path to image folder (e.g., /path/to/train2014)")
    parser.add_argument("--ann_path", default='dataset/vqc/train', type=str, required=False,
                        help="Path to annotation JSON (e.g., /path/to/mdetr_periapical_annotations.json)")
    args = parser.parse_args()

    main(args)
