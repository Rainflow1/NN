import random

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
import os
from scipy.ndimage import gaussian_filter
import cv2


class UCFDataset(Dataset):
    def __init__(self, dataset_path, crop=512, mode="train"): # albo 256
        self.root = dataset_path
        self.crop = crop
        self.mode = mode
        files = []
        for file in os.listdir(dataset_path):
            if os.path.isfile(os.path.join(dataset_path, file)) and file.endswith('.jpg'):
                files.append(file)
        self.files = files
        self.samples = len(files)

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        gt_path = img_path.replace(".jpg", ".npy")

        img = Image.open(img_path).convert("RGB")
        gt = np.load(gt_path)

        if self.mode == "train":
            return self.train_transform(img, gt)
        elif self.mode == "val":
            return F.to_tensor(img), len(gt), self.files[index]
        else:
            width, height = img.size
            idx_mask = (gt[:, 0] >= 0) * (gt[:, 0] <= width) * (gt[:, 1] >= 0) * (gt[:, 1] <= height)
            gt = gt[idx_mask]
            points = np.zeros((height, width))
            for i in range(len(gt)):
                points[int(gt[i, 1]), int(gt[i, 0])] = 1.0

            points = gaussian_filter(points, sigma=8)
            points = cv2.resize(points, (int(width / 8), int(height / 8)), interpolation=cv2.INTER_CUBIC) * 64

            return F.to_tensor(img), len(gt), points

    def train_transform(self, img, points):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)
        assert st_size >= self.crop
        assert len(points) > 0
        i, j, h, w = self.img_crop(ht, wd, self.crop)
        img = F.crop(img, i, j, h, w)
        nearest_dis = np.clip(points[:, 2], 4.0, 128.0)

        points_left_up = points[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = points[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = self.cal_innner_area(j, i, j+w, i+h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        points = points[mask]
        points = points[:, :2] - [j, i]  # change coodinate
        if len(points) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                points[:, 0] = w - points[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return F.to_tensor(img), torch.from_numpy(points.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size

    def img_crop(self, width, height, crop):
        anchor_x = width - crop
        anchor_y = height - crop
        return random.randint(0, anchor_y), random.randint(0, anchor_x), crop, crop

    def cal_innner_area(self, c_left, c_up, c_right, c_down, bbox):
        inner_left = np.maximum(c_left, bbox[:, 0])
        inner_up = np.maximum(c_up, bbox[:, 1])
        inner_right = np.minimum(c_right, bbox[:, 2])
        inner_down = np.minimum(c_down, bbox[:, 3])
        inner_area = np.maximum(inner_right - inner_left, 0.0) * np.maximum(inner_down - inner_up, 0.0)
        return inner_area

if __name__ == "__main__":
    tset = UCFDataset("./dataset/val", mode="val")
    print()

