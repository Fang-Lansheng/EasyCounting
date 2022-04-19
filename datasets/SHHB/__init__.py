import os
import cv2
import glob
import torch
import numpy as np

from datasets import BaseLoader
from misc.utilities import resize_dot_map


class SHHB(BaseLoader):
    def __init__(self, data_dir, transforms, crop_size=400, scale=8, mode='train'):
        super(SHHB, self).__init__(data_dir, transforms, crop_size, scale, mode)
        self.get_file_path()

    def get_file_path(self):
        sub_data_dir = 'train_data' if self.mode == 'train' else 'test_data'
        img_dir_path = os.path.join(self.data_dir, sub_data_dir, 'images')

        path_list = glob.glob(os.path.join(img_dir_path, '*.jpg'))
        path_list = sorted(path_list, key=lambda s: int(s.split('IMG_')[1][:-4]))

        for img_path in path_list:
            dot_path = img_path.replace('.jpg', '.h5').replace('images', 'dot_map')
            den_path = dot_path.replace('dot_map', 'density')

            self.img_path_list.append(img_path)
            self.dot_path_list.append(dot_path)
            self.den_path_list.append(den_path)

    def train_transform(self):
        self.random_resize()
        self.random_crop()
        self.random_flip()
        self.random_gamma()

        self.img = self.transforms(self.img)

        self.den = cv2.resize(
            self.den, (self.crop_size // self.scale, self.crop_size // self.scale),
            interpolation=cv2.INTER_LINEAR)
        self.den *= self.scale**2
        self.den = self.den[np.newaxis, :, :]
        self.den = torch.from_numpy(self.den).float()

        points = np.array(np.where(self.dot > 0)).transpose()
        permutation = [1, 0]
        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))
        points[:] = points[:, idx]
        points = torch.from_numpy(points).float()

        self.dot = resize_dot_map(self.dot, 1 / self.scale, 1 / self.scale)
        self.dot = self.dot[np.newaxis, :, :]
        self.dot = torch.from_numpy(self.dot).float()

        return self.img, self.dot, self.den, points
