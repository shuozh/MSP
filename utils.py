import os
import random
from PIL import Image
import torch
import numpy as np
import torchvision.transforms.functional as Ft
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class llf_dataset(Dataset):
    def __init__(self, args, is_train=True):
        self.root_dir = args.data_dir
        self.n_view = args.n_view

        self.is_train = is_train

        self.random = random


        print("dataset: ", args.dataset)

        if self.is_train:
            self.data_dir = os.path.join(self.root_dir, 'train', args.dataset)
            self.gt_dir = os.path.join(self.root_dir, 'train/1')
        else:
            self.data_dir = os.path.join(self.root_dir, 'test', args.dataset)
            self.gt_dir = os.path.join(self.root_dir, 'test/1')

        self.imgs = sorted(os.listdir(self.data_dir))
        self.imgs_gt = sorted(os.listdir(self.gt_dir))

        self.img_list = []
        self.gt_list = []
        print("Loading data.")
        for idx, _ in enumerate(self.imgs):
            self.img_list.append(np.load(os.path.join(self.data_dir, self.imgs[idx])))
            self.gt_list.append(np.load(os.path.join(self.gt_dir, self.imgs_gt[idx])))
        print("DONE.")


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        self.img = self.img_list[idx]
        self.img_gt = self.gt_list[idx]

        # 9 * 9  --->  7 * 7
        if self.n_view % 2 != 0:
            start = int((9 - self.n_view) / 2)
            end = -start if start != 0 else self.n_view
        else:
            start = int((9 - self.n_view) // 2)
            end = -np.ceil((9 - self.n_view) / 2).astype(np.int8)

        data = self.img[start:end, start:end]
        gt = self.img_gt[start:end, start:end]
        _, _, h, w, _ = data.shape

        data = torch.from_numpy(data).reshape(self.n_view * self.n_view, h, w, 3).float()
        gt = torch.from_numpy(gt).reshape(self.n_view * self.n_view, h, w, 3).float()
        data = data[:, :h//4 * 4, :w//4 * 4, :]
        gt = gt[:, :h//4 * 4, :w//4 * 4, :]

       
        data, gt = data / 255, gt / 255

        data, gt = data.permute([0, 3, 1, 2]), gt.permute([0, 3, 1, 2])

        return data, gt

