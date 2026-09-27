from ultralytics import YOLO
import cv2
import torch

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = YOLO(r"C:/Users/Dukare sakshi kisan/Desktop/crop-and-weed-main/crop-and-weed-main/runs/detect/train-4/weights/best.pt")

# image path
img_path = r"c:/Users/Dukare sakshi kisan/Desktop/crop-and-weed-main/crop-and-weed-main/input/crop-and-weed-detection-data-with-bounding-boxes/versions/1/agri_data/data/agri_0_113.jpeg"

# read image
img = cv2.imread(img_path)

# run prediction (NO show=True)
results = model(img_path)

# class names
names = model.names

# loop through detections
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]}: {conf:.2f}"

        # draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # put label
        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

# show image manually
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save image manually
cv2.imwrite("output.jpg", img)