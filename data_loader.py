import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import cv2


# ================= 数据增强 =================
def cv_random_flip(img, label):
    if random.randint(0, 1) == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomRotation(image, label):
    if random.random() > 0.8:
        angle = np.random.randint(-15, 15)
        image = image.rotate(angle, Image.BICUBIC)
        label = label.rotate(angle, Image.BICUBIC)
    return image, label


def randomCrop(image, label):
    border = 30
    w, h = image.size
    cw = np.random.randint(w - border, w)
    ch = np.random.randint(h - border, h)
    region = (
        (w - cw) >> 1, (h - ch) >> 1,
        (w + cw) >> 1, (h + ch) >> 1
    )
    return image.crop(region), label.crop(region)


# ================= Dataset =================
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize=384, is_train=False, datasetname_=''):
        self.trainsize = trainsize
        self.is_train = is_train
        self.datasetname = datasetname_

        # 正确路径拼接
        self.images = sorted([
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        self.gts = sorted([
            os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        assert len(self.images) == len(self.gts), \
            f"Image/Gt 数量不一致: {len(self.images)} vs {len(self.gts)}"

        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

        self.edge_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path = self.images[index]
        gt_path = self.gts[index]

        # ✅ 防御式检查（非常重要）
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[Image Missing] {img_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"[GT Missing] {gt_path}")

        image = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")

        if self.is_train and self.datasetname == 'CrackSeg9k':
            image, gt = cv_random_flip(image, gt)
            image, gt = randomRotation(image, gt)
            image, gt = randomCrop(image, gt)

        gt_np = np.array(gt)
        edge = cv2.Canny(gt_np, 100, 200)
        edge = cv2.dilate(edge, np.ones((5, 5), np.uint8), iterations=1)

        image = self.img_transform(image)
        gt = self.gt_transform(Image.fromarray(gt_np))
        edge = self.edge_transform(Image.fromarray(edge))

        return {
            'image': image,
            'label': gt,
            'edge': edge
        }

    def __len__(self):
        return len(self.images)


# ================= DataLoader =================
def get_loader(image_root, gt_root, batchsize, trainsize,
               is_train=False, shuffle=True, num_workers=16,
               pin_memory=True, datasetname_=''):

    dataset = SalObjDataset(
        image_root=image_root,
        gt_root=gt_root,
        trainsize=trainsize,
        is_train=is_train,
        datasetname_=datasetname_
    )

    return data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
