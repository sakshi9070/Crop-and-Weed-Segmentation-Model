import numpy as np
import pandas as pd
import torch
import glob
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- DATA PATH ----------------
jpegs = glob.glob(
    r'c:/Users/Dukare sakshi kisan/Desktop/crop-and-weed-main/crop-and-weed-main/input/crop-and-weed-detection-data-with-bounding-boxes/versions/1/agri_data/data/*.jpeg'
)

txts = [i.rsplit('.', 1)[0] + '.txt' for i in jpegs]

transformer = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# ---------------- DATASET ----------------
class GenData(Dataset):
    def __init__(self, jpegs, txts, transformer):
        self.jpegs = jpegs
        self.txts = txts
        self.transformer = transformer

    def __getitem__(self, item):
        img = Image.open(self.jpegs[item]).convert('RGB')
        img = self.transformer(img)

        df = pd.read_csv(self.txts[item], header=None, sep=' ')
        df.columns = ['label', 'x_cen', 'y_cen', 'w', 'h']

        # convert YOLO -> bbox
        df['xmin'] = (df['x_cen'] - df['w'] / 2) * 512
        df['ymin'] = (df['y_cen'] - df['h'] / 2) * 512
        df['xmax'] = (df['x_cen'] + df['w'] / 2) * 512
        df['ymax'] = (df['y_cen'] + df['h'] / 2) * 512

        bbox = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        label = df['label'].values.tolist()

        bbox = torch.tensor(bbox, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        target = {
            'boxes': bbox,
            'labels': label
        }

        return img, target

    def __len__(self):
        return len(self.jpegs)


dataset = GenData(jpegs, txts, transformer)

train_len = int(len(dataset) * 0.7)
test_len = len(dataset) - train_len

dataset_train, dataset_test = random_split(dataset, [train_len, test_len])

# ---------------- DATALOADER ----------------
def detection_collate(x):
    return tuple(zip(*x))

dl_train = DataLoader(dataset_train, batch_size=1, shuffle=True, collate_fn=detection_collate)
dl_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=detection_collate)

# ---------------- VISUALIZATION ----------------
class_idx1 = {0: 'crop', 1: 'weed'}

def train_img_show(dl_train):
    img, label = next(iter(dl_train))

    img_sample = np.transpose(np.array(img[0]), (1, 2, 0)) * 255
    img_sample = img_sample.astype(np.uint8).copy()

    boxes = label[0]['boxes'].numpy()
    labels = label[0]['labels'].numpy()

    for i in range(len(labels)):
        x1, y1, x2, y2 = map(int, boxes[i])
        text = class_idx1.get(labels[i], "unknown")

        cv2.rectangle(img_sample, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(img_sample, text, (x1, y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)

    plt.imshow(img_sample)
    plt.show()

train_img_show(dl_train)

# ---------------- MODEL ----------------
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=3)
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0001)

# ---------------- TRAINING ----------------
def train_one_epoch(model, optimizer, dl_train, dl_test, device, epochs):

    for epoch in range(epochs):
        model.train()

        loss_epoch = []
        iou_epoch = []

        for images, targets in tqdm(dl_train):

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_epoch.append(losses.item())

        # ---------------- VALIDATION ----------------
        model.eval()
        test_loss = []

        with torch.no_grad():
            for images, targets in dl_test:

                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                test_loss.append(losses.item())

        # cleanup (safe)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\nEpoch {epoch+1}")
        print("Train Loss:", np.mean(loss_epoch))
        print("Test Loss:", np.mean(test_loss))

        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")


# ---------------- TRAIN ----------------
train_one_epoch(model, optimizer, dl_train, dl_test, device, 15)

# ---------------- INFERENCE ----------------
model.eval()

names = {0: 'crop', 1: 'weed'}

img_path = jpegs[0]
src_img = plt.imread(img_path)

img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(device)

with torch.no_grad():
    out = model([img_tensor])

boxes = out[0]['boxes'].cpu().numpy().astype(int)
labels = out[0]['labels'].cpu().numpy()
scores = out[0]['scores'].cpu().numpy()

for i in range(len(boxes)):
    if scores[i] >= 0.8:
        x1, y1, x2, y2 = boxes[i]
        label_name = names.get(labels[i], "unknown")

        cv2.rectangle(src_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(src_img, label_name, (x1, y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)

plt.imshow(src_img)
plt.show()

import numpy as np
import torch
import torchvision

def evaluate_iou(model, dataloader, device):
    model.eval()

    iou_scores = []

    with torch.no_grad():
        for images, targets in dataloader:

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for i in range(len(outputs)):

                pred_boxes = outputs[i]['boxes']
                true_boxes = targets[i]['boxes']

                if len(pred_boxes) == 0 or len(true_boxes) == 0:
                    continue

                iou_matrix = torchvision.ops.box_iou(true_boxes, pred_boxes)

                # best prediction per ground truth
                best_iou = torch.max(iou_matrix, dim=1)[0]

                iou_scores.extend(best_iou.cpu().numpy())

    return np.mean(iou_scores) if len(iou_scores) > 0 else 0

train_iou = evaluate_iou(model, dl_train, device)
test_iou = evaluate_iou(model, dl_test, device)

print("Train IoU:", train_iou)
print("Test IoU:", test_iou)

import cv2
import matplotlib.pyplot as plt
import torch

def predict_image(model, image_path, device, threshold=0.8):

    model.eval()

    names = {0: 'crop', 1: 'weed'}

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = torch.from_numpy(img_rgb / 255.).permute(2, 0, 1).float().to(device)

    with torch.no_grad():
        output = model([img_tensor])

    boxes = output[0]['boxes'].cpu().numpy().astype(int)
    labels = output[0]['labels'].cpu().numpy()
    scores = output[0]['scores'].cpu().numpy()

    for i in range(len(boxes)):
        if scores[i] >= threshold:
            x1, y1, x2, y2 = boxes[i]
            label_name = names.get(labels[i], "unknown")

            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb, f"{label_name} {scores[i]:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)

    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()


predict_image(
    model,
    "c:/Users/Dukare sakshi kisan/Desktop/crop-and-weed-main/crop-and-weed-main/input/crop-and-weed-detection-data-with-bounding-boxes/versions/1/agri_data/data/agri_0_113.jpeg",
    device
)