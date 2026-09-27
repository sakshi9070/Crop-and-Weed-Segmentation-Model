import numpy as np
import pandas as pd
import torch
import glob
from torch.utils.data import Dataset,DataLoader,random_split
from PIL import Image
import torchvision
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

jpegs = glob.glob('c:/Users/Dukare sakshi kisan/Desktop/crop-and-weed-main/crop-and-weed-main/input/crop-and-weed-detection-data-with-bounding-boxes/versions/1/agri_data/data/*.jpeg')
txts = [i.rsplit('.',maxsplit=1)[0]+'.txt' for i in jpegs]
jpegs[0]
transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
class GenData(Dataset):
    def __init__(self,jpegs,txts,transformer):
        self.jpegs = jpegs
        self.txts = txts
        self.transformer = transformer
    def __getitem__(self, item):
        img = Image.open(jpegs[item]).convert('RGB')
        img = transformer(img)
        df = pd.read_csv(txts[item], header=None, sep=' ')
        df.columns = ['label', 'x_cen', 'y_cen', 'w', 'h']
        df['xmin'] = (df['x_cen'] - df['w'] / 2) * 512
        df['ymin'] = (df['y_cen'] - df['h'] / 2) * 512
        df['xmax'] = (df['x_cen'] + df['w'] / 2) * 512
        df['ymax'] = (df['y_cen'] + df['h'] / 2) * 512
        bbox = np.array(df.iloc[:,5:]).tolist()
        label = np.array(df.iloc[:, 0]).squeeze().tolist()
        bbox = torch.tensor(bbox,dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.int64).reshape(-1,)
        target = {}
        target['boxes'] = bbox
        target['labels'] = label
        return img,target
    def __len__(self):
        return len(self.jpegs)

dataset = GenData(jpegs,txts,transformer)
train_len = int(dataset.__len__()*0.7)
test_len = int(dataset.__len__()-train_len)
dataset_train,dataset_test = random_split(dataset,[train_len,test_len])

def detection_collate(x):
    return list(tuple(zip(*x)))
dl_train = DataLoader(dataset_train,1,shuffle=True,collate_fn=detection_collate)
dl_test = DataLoader(dataset_test,1,shuffle=True,collate_fn=detection_collate)
class_idx1={0:'crop',1:'weed'}
def train_img_show(dl_train,class_idx1):
    img,label = next(iter(dl_train))

    img_sample = np.transpose(np.array(img[0]),(1,2,0))*255
    img_sample = img_sample.astype(np.uint8)
    img_sample = img_sample.copy()
    box1 = label[0]['boxes'].numpy()
    label1 = label[0]['labels'].numpy()

    for i in range(len(label1)):
        x1,y1,x2,y2 = int(box1[i][0]),int(box1[i][1]),int(box1[i][2]),int(box1[i][3])
        label = str(class_idx1[label1[i]])
        cv2.rectangle(img_sample,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.putText(img_sample,text=label,fontScale=0.5,fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness=1,lineType=cv2.LINE_AA,color=(0,0,255),org=(x1,y1+10))
    plt.imshow(img_sample)
train_img_show(dl_train,class_idx1)
model=torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.Adam(params,lr=0.0001)
loss_list = []

def train_one_epoch(model, optimizer, dl_train, dl_test, device, epochs):
    for epoch in range(epochs):
        loss_epoch = []
        iou_epoch = []
        for images, targets in tqdm(dl_train):
            model.train()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                loss_epoch.append(losses.cpu().numpy())
                label = targets
                try:
                    pred = model(images)
                    for i in range(len(pred)):
                        pic_boxes = pred[i]['boxes']
                        label_boxes = label[i]['boxes']
                        iou_tensor = torchvision.ops.box_iou(label_boxes, pic_boxes)
                        iou_total = np.mean(torch.max(iou_tensor, dim=1)[0].cpu().numpy())
                        iou_epoch.append(iou_total)

                except:
                    continue
        test_loss_epoch = []
        test_iou_epoch = []
        gc.collect()
        torch.cuda.empty_cache()
        for images, targets in tqdm(dl_test):
            model.train()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            with torch.no_grad():
                model.eval()
                test_loss_epoch.append(losses.cpu().numpy())
                label = list(targets)
                try:
                    pred = model(images)
                    for i in range(len(pred)):
                        pic_boxes = pred[i]['boxes']
                        label_boxes = label[i]['boxes']
                        iou_tensor = torchvision.ops.box_iou(label_boxes, pic_boxes)
                        iou_total = np.mean(torch.max(iou_tensor, dim=1)[0].cpu().numpy())
                        test_iou_epoch.append(iou_total)

                except:
                    continue

        mIou = np.mean(iou_epoch)
        epochloss = np.mean(loss_epoch)
        test_mIou = np.mean(test_iou_epoch)
        test_epochloss = np.mean(test_loss_epoch)
        static_dict = model.state_dict()
        torch.save(static_dict,'./{}test_mIou_{}test_loss{}.pth'.format(epoch,round(test_mIou,3),round(test_epochloss,3)))

        print('\n',
              'epoch:', epoch,
              'epoch_mIou:', mIou,
              'epoch_loss:', epochloss,
              'test_epoch_mIou:', test_mIou,
              'test_epoch_loss:', test_epochloss)
        gc.collect()
        torch.cuda.empty_cache()


train_one_epoch(model, optimizer, dl_train,dl_test,device, 15)
model.eval()
names = {'0': 'crop', '1': 'weed'}
src_img = plt.imread(r"c:/Users/Dukare sakshi kisan/Desktop/crop-and-weed-main/crop-and-weed-main/input/crop-and-weed-detection-data-with-bounding-boxes/versions/1/agri_data/data/agri_0_113.jpeg")
img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().device()

out = model(torch.unsqueeze(img_tensor,dim=0))
boxes = out[0]['boxes'].cpu().detach().numpy().astype(int)
labels = out[0]['labels'].cpu().detach().numpy()
scores = out[0]['scores'].cpu().detach().numpy()
for idx in range(boxes.shape[0]):
    if scores[idx] >= 0.8:
        x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
        name = names.get(str(labels[idx].item()))
        cv2.rectangle(src_img,(x1,y1),(x2,y2),(255,0,0),thickness=1)
        cv2.putText(src_img, text=name, org=(x1, y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

plt.imshow(src_img)