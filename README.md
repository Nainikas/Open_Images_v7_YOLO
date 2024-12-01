# Open_Images_v7_YOLO

---

# OpenImages + FiftyOne + YOLOv8 Training and Evaluation

This repository contains the complete workflow for training a YOLOv8 model using OpenImages V7 dataset, leveraging FiftyOne for dataset management and YOLOv8 for object detection. This workflow is tailored for detecting classes: **Car**, **Traffic Light**, and **Traffic Sign**, and aims to achieve high-precision object detection.

## Access the Notebook
The Colab notebook for the entire workflow can be accessed here:  
[OpenImages FiftyOne YOLOv8 Notebook](https://colab.research.google.com/drive/1wAuIkPAmnblV92GexjxM1iUDSBqOcaFN?usp=sharing).

---

## Project Overview
- **Dataset Source**: OpenImages V7
- **Classes Detected**: 
  - Car
  - Traffic Light
  - Traffic Sign
- **Tools Used**:
  - FiftyOne for dataset curation and management.
  - YOLOv8 for training and evaluation.
  - Google Colab for leveraging high computational resources (GPU).
- **Training Phases**:
  1. Dataset Filtering & Preprocessing
  2. Model Training
  3. Evaluation & Inference
- **Output Metrics**:
  - Precision: ~0.69
  - Recall: ~0.57
  - mAP@50: ~0.63
  - mAP@50-95: ~0.48

---

## Features
### Dataset Preprocessing
- Utilized FiftyOne for filtering classes of interest and exporting the dataset in YOLOv5 format.
- Verified annotations and ensured no missing labels for robust model training.

### Training Configuration
- **Model Used**: YOLOv8n (Nano) pre-trained weights.
- **Training Parameters**:
  - Epochs: 100
  - Batch Size: 16
  - Image Size: 640
  - Optimizer: SGD with momentum 0.9
  - Augmentations: Blur, CLAHE, ToGray, and Albumentations.
- **Training Logs**:
  - Precision stabilized around 0.69.
  - mAP@50 showed consistent improvement across epochs.

### Evaluation
- Validation Dataset Split: 20% of the dataset.
- Metrics Evaluated:
  - Precision, Recall, and mAP scores across classes.
- Observations:
  - The model showed a balanced performance across classes, with slightly higher precision for **Car**.
  - Moderate recall for **Traffic Lights** and **Traffic Signs**.

### Inference
- The model was tested on unseen data and predictions were visualized for performance validation.

---

## Metrics Breakdown (After 100 Epochs)
- **Precision (P)**: 0.69
- **Recall (R)**: 0.57
- **mAP@50**: 0.63
- **mAP@50-95**: 0.48
- Class-Wise Performance:
  - **Car**: Highest Precision (~0.72)
  - **Traffic Light**: Moderate (~0.63)
  - **Traffic Sign**: Challenging (~0.48)

---

## Visualizations
Sample predictions are saved under `/content/yolo_project/test_predictions2/` directory. You can visualize the bounding boxes and class predictions using the following snippet:

```python
import os
from IPython.display import Image, display

prediction_dir = "/content/yolo_project/test_predictions2"
for img_file in os.listdir(prediction_dir):
    if img_file.endswith(".jpg"):
        display(Image(filename=os.path.join(prediction_dir, img_file)))
```

---

## How to Run
### 1. Dataset Preparation
```bash
pip install fiftyone
```
Download and filter the OpenImages dataset using FiftyOne:

```python
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    splits=("train", "validation"),
    classes=["Car", "Traffic Light", "Traffic Sign"],
    label_types=["detections"],
)
```

### 2. Model Training
Train the YOLOv8 model with the following configuration:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # YOLOv8 nano model
model.train(
    data="/content/dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # Use GPU
    project="yolo_project",
    name="training_run"
)
```

### 3. Evaluate the Model
```python
metrics = model.val(
    data="/content/dataset/data.yaml",
    split="test",
    project="yolo_project",
    name="test_evaluation"
)
print(metrics)
```

---

## Improving Performance
- **Increase Epochs**: Extend training epochs to 200 or more to allow better convergence.
- **Advanced Augmentations**: Include techniques like Mosaic and CutMix.
- **Custom Backbone**: Explore larger YOLOv8 models or custom architectures.

---

## Feedback and Contributions
Feel free to contribute by raising issues or submitting pull requests to this repository. Your feedback is invaluable for refining this workflow!

---
