The code snippet below is used to train YOLO models and was featured in the video at the following link:


!pip install -U ultralytics opencv-python


import sys, torch
print("PY:", sys.executable)
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


from pathlib import Path
# Path to your dataset
DATASET_REDraspberry = Path(r"C:/Users/HARDPC/Desktop/REDraspberries/dataset_REDraspberry")
# Checking the date structure
for p in [
    DATASET_REDraspberry,
    DATASET_REDraspberry / "images/train",
    DATASET_REDraspberry / "images/val",     
    DATASET_REDraspberry / "images/test", 
    DATASET_REDraspberry / "labels/train",
    DATASET_REDraspberry / "labels/val",
    DATASET_REDraspberry / "labels/test"
]:
    print(p, "→", "OK ✅" if p.exists() else "❌ not exist")


data_yaml = """
path: C:/Users/HARDPC/Desktop/REDraspberries/dataset_REDraspberry
train: images/train
val: images/val
test: images/test
names:
  0: red
"""
with open("data_REDraspberries.yaml", "w", encoding="utf-8") as f:
    f.write(data_yaml)
print("✔ Save data_REDraspberries.yaml")

from ultralytics import YOLO
# Loading the base model
model = YOLO("yolo11s.pt")
# Training
results = model.train(
    data="data_REDraspberries.yaml",         
    epochs=200,
    batch=32, # optimized for RTX 5080
    imgsz=640,
    optimizer='SGD',
    momentum=0.937,
    weight_decay=0.0005,
    lr0=0.01,
    lrf=0.01,
    seed=0,
    augment=True,
    workers=8,  # optimized for Ryzen 9 9950X
    patience=0,
    device='cuda',
    project="runs/train_REDraspberries",     # Save location
    name="yolov11s_REDraspberries",           # Session name
