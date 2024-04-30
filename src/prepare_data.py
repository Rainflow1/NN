import random
import cv2
import os
import numpy as np
from scipy.io import loadmat
from PIL import Image
import tqdm
from itertools import islice


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio

def load_files(img_path, gt_path):
    img = Image.open(img_path)
    img_w, img_h = img.size
    points = loadmat(gt_path)['annPoints'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= img_w) * (points[:, 1] >= 0) * (points[:, 1] <= img_h)
    points = points[idx_mask]
    img_h, img_w, rr = cal_new_size(img_h, img_w, 512, 2048)
    im = np.array(img)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (img_w, img_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points

def find_dis(point):
    square = np.sum(point*point, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis


if __name__ == "__main__":

    dataset_path = "C:\\Users\\Macie\\Downloads\\UCF-QNRF_ECCV18"
    new_path = "./dataset"

    for phase in ["train", "test"]:
        subdir = os.path.join(dataset_path, phase)

        files = []
        for file in os.listdir(os.path.join(dataset_path, phase)):
            if file.endswith(".jpg"):
                files.append(file)

        random.shuffle(files)

        if phase == "train":
            sub_phases = ["train", "val"]
            phase_files = {"train": [f.replace("\n", "") for f in open("train.txt")], "val": []}
            phase_files["val"] = list(set(files) - set(phase_files["train"]))

            for sub_phase in sub_phases:
                subphase_dir = os.path.join(new_path, sub_phase)

                if not os.path.exists(subphase_dir):
                    os.makedirs(subphase_dir)
                for file in os.listdir(subphase_dir):
                    os.remove(os.path.join(subphase_dir, file))

                for file in tqdm.tqdm(phase_files[sub_phase]):
                    img_path = os.path.join(dataset_path, phase, file)
                    gt_path = img_path.replace(".jpg", "_ann.mat")

                    img, gt = load_files(img_path, gt_path)
                    if sub_phase == 'train':
                        dis = find_dis(gt)
                        gt = np.concatenate((gt, dis), axis=1)

                    img_new_path = os.path.join(subphase_dir, file)
                    gt_new_path = img_new_path.replace(".jpg", ".npy")
                    img.save(img_new_path)
                    np.save(gt_new_path, gt)
        else:
            subphase_dir = os.path.join(new_path, phase)

            if not os.path.exists(subphase_dir):
                os.makedirs(subphase_dir)
            for file in os.listdir(subphase_dir):
                os.remove(os.path.join(subphase_dir, file))

            for file in tqdm.tqdm(files):
                img_path = os.path.join(dataset_path, phase, file)
                gt_path = img_path.replace(".jpg", "_ann.mat")

                img, gt = load_files(img_path, gt_path)

                img_new_path = os.path.join(subphase_dir, file)
                gt_new_path = img_new_path.replace(".jpg", ".npy")
                img.save(img_new_path)
                np.save(gt_new_path, gt)
