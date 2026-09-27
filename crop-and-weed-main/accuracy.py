from ultralytics import YOLO
import torch

# device (CPU/GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# load trained model
model = YOLO(r"C:/Users/Dukare sakshi kisan/Desktop/crop-and-weed-main/crop-and-weed-main/runs/detect/train-4/weights/best.pt")

# image path
img_path = r"c:/Users/Dukare sakshi kisan/Desktop/crop-and-weed-main/crop-and-weed-main/input/crop-and-weed-detection-data-with-bounding-boxes/versions/1/agri_data/data/agri_0_113.jpeg"

# ✅ prediction + display
results = model(img_path, show=True)

# ✅ extract results
for r in results:
    boxes = r.boxes.xyxy
    conf = r.boxes.conf
    cls = r.boxes.cls

    print("\nDetections:")
    print("Boxes:", boxes)
    print("Confidence:", conf)
    print("Classes:", cls)

# ✅ save output image
model.predict(source=img_path, save=True)

# ✅ (optional) evaluate model
# comment this if you don’t want validation every run
metrics = model.val()
print("\nValidation Metrics:", metrics)