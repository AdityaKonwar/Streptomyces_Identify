# Streptomyces_Identify

A basic machine learning model created using a pre-trained YOLOv11 model. The model was trained for the identification of feature rich Actinobacteria plate culture picture and differentiate specifically based on the rough matt-like textures of the colonies. The input images were pre-processed for better texture based differentiation using local_binary_pattern (LBP) for texture analysis and feature extraction, CLAHE - for contrast enhancement, and HED - or Holistically-Nested Edge Detection, which is a deep learning-based method used for image pre-processing, specifically for edge detection.

# **How to load the model:**

The model can be used from the ultralytics library and then loading the .pt file from your computer or cloud. For example:

`from ultralytics import YOLO
model = YOLO("your input path")  # e.g., 'streptomyces_model.pt'`

*The trained model file is already uploaded to this page for your use. Just download and specify the path in your VS code or google colab.*

# **Model details:**

The model was trained using images that were processed for better texture based differentiation using local_binary_pattern (LBP) for texture analysis and feature extraction, CLAHE - for contrast enhancement, and HED - or Holistically-Nested Edge Detection, which is a deep learning-based method used for image pre-processing, specifically for edge detection.

# **Data collection and classification:**
Data (images) were collected in two folders i.e. Streptomyces and Non-Streptomyces. For images, confirmed plate pictures from databases such as BacDive, Shuttershock, etc. were collected where the plate picture was correlated with sequence data which confirms to which genus the bacteria belongs to.
Approximately 200 images were collected for each set and were pre-processed for enhancing its contrast, texture and edge detection.
The data was divided randomly into training, validation, and test for each set.
`classes = ['Streptomyces', 'non_streptomyces']
split_ratio = 0.8  # 80% train, 20% val`

The processed images were then combined into a single image .combo and then these images were randomly split into 'train' and 'val' set.

First load these libraries:
`import os
import shutil
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt`
`

Clear/Create target folder structure
`for split in ['train', 'val']:
    for cls in classes:
        dir_path = os.path.join(yolo_base, split, cls)
        os.makedirs(dir_path, exist_ok=True)
        # Optional: clear old files
        for f in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, f))`

Copy only *_combo.* images into train/val
`for cls in classes:
    src_dir = os.path.join(processed_base, cls)
    images = [f for f in os.listdir(src_dir)]
    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for fname in train_imgs:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(yolo_base, 'train', cls, fname)
        shutil.copy2(src_path, dst_path)

    for fname in val_imgs:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(yolo_base, 'val', cls, fname)
        shutil.copy2(src_path, dst_path)

print("✅ YOLO dataset ready with combo images only.")`

# **Training YOLO model:**
Now just load a pre-trained YOLO version and then use it for training with your dataset. Just specify the path and its epochs, batch, and hardware specifications.
`!pip install -U ultralytics
from ultralytics import YOLO`
`model = YOLO('yolo11s-cls.pt')  # Pretrained YOLOv11 classification model

model.train(
    data=DATASET_PATH,
    imgsz=512,
    epochs=100,
    batch=16,
    device=0  # GPU (if available)
)`
The model, while training, used the following hardware features:
Ultralytics 8.3.145 🚀 Python-3.11.12 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)
