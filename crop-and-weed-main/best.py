from ultralytics import YOLO
import os
import shutil
import random

ROOT = r"c:/Users/Dukare sakshi kisan/Desktop/crop-and-weed-main/crop-and-weed-main/input/crop-and-weed-detection-data-with-bounding-boxes/versions/1/agri_data/data"

DEST = r"c:/Users/Dukare sakshi kisan/Desktop/yolo_dataset"

IMG_DIR = os.path.join(DEST, "images")
LBL_DIR = os.path.join(DEST, "labels")

def move_files(file_list, subset):
    for img in file_list:

        txt = os.path.splitext(img)[0] + ".txt"

        src_img = os.path.join(ROOT, img)
        src_lbl = os.path.join(ROOT, txt)

        dst_img = os.path.join(IMG_DIR, subset, img)
        dst_lbl = os.path.join(LBL_DIR, subset, txt)

        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.copy(src_img, dst_img)
            shutil.copy(src_lbl, dst_lbl)

# folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(IMG_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(LBL_DIR, split), exist_ok=True)

# images
jpg_files = [f for f in os.listdir(ROOT) if f.endswith(".jpeg")]

random.seed(42)
random.shuffle(jpg_files)

split_idx = int(len(jpg_files) * 0.8)
train_files = jpg_files[:split_idx]
val_files = jpg_files[split_idx:]

move_files(train_files, "train")
move_files(val_files, "val")

# YAML
data_yaml = '''
train: c:/Users/Dukare sakshi kisan/Desktop/yolo_dataset/images/train
val: c:/Users/Dukare sakshi kisan/Desktop/yolo_dataset/images/val

nc: 2
names: ['crop', 'weed']
'''

yaml_path = r"c:/Users/Dukare sakshi kisan/Desktop/yolo_dataset/data.yaml"

with open(yaml_path, 'w') as f:
    f.write(data_yaml)

# TRAIN
model = YOLO("yolov8n.pt")

model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=8
)

