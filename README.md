# Multimodal Vision-Language Models on Smart Hospital

**Task**: Referring Expression Comprehension (RefExp) for detecting periapical inflammation in panoramic dental X-ray images.

## 🚑 Overview

This project applies a multimodal vision-language model to detect regions of periapical inflammation in dental panoramic X-ray images based on textual descriptions. The goal is to support smart hospital systems by improving automated image interpretation using referring expressions (e.g., "inflammation near the lower right molar").

## 📂 Dataset

* **Format**: COCO-style with custom annotations for referring expressions and bounding boxes.
* **Image type**: Panoramic dental X-rays.
* **Annotations**:

  * `image_id`
  * `file_name`
  * `annotations`: bounding boxes (x, y, width, height), category, and natural language expression.

## 🔧 Setup

```bash
git clone https://github.com/your-username/multimodal-smart-hospital.git
cd multimodal-smart-hospital
conda create -n refexp python=3.9
conda activate refexp
pip install -r requirements.txt
```

## 🧪 Demo: Run Inference

```bash
python main.py \
  --dataset_config configs/refexp_panorama.yaml \
  --coco_path /path/to/dataset \
  --output-dir outputs \
  --text_encoder_type roberta-base \
  --backbone resnet50 \
  --num_queries 50 \
  --masks \
  --eval
```

## 🧠 Model

We use a DETR-based transformer model that jointly processes visual features and encoded text to predict bounding boxes that correspond to the described pathological region.
