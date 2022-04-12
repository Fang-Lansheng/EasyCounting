import cv2
import h5py
import glob
import torch
import random
import numpy as np
import os.path as osp

from PIL import Image
from datasets import BaseLoader


class UCSD(BaseLoader):
    def __init__(self, data_dir, transforms, crop_size=400, scale=8, mode='train'):
        super(UCSD, self).__init__(data_dir, transforms, crop_size, scale, mode)

        self.time_step = scale

        self.get_file_path()

    def __getitem__(self, item):
        imgs, dens = self.load_files(item)

        for i, [img, den] in enumerate(zip(imgs, dens)):
            # imgs[i] = torch.from_numpy(img).float().unsqueeze(0)
            imgs[i] = self.transforms(img)
            dens[i] = torch.from_numpy(den).float().unsqueeze(0)

        imgs = torch.stack(imgs)
        dens = torch.stack(dens)

        if self.mode == 'train':
            # random horizontal flip
            if random.random() < 0.5:
                imgs = torch.flip(imgs, dims=[-1])
                dens = torch.flip(dens, dims=[-1])

            return imgs, dens
        elif self.mode == 'test':
            return imgs, dens

    def __len__(self):
        return int(len(self.img_path_list) / self.scale)

    def get_file_path(self):
        sub_data_dir = 'train_data' if self.mode == 'train' else 'test_data'
        img_dir = osp.join(self.data_dir, sub_data_dir, 'imgs')

        img_path_list = glob.glob(osp.join(img_dir, '*.png'))
        img_path_list = sorted(img_path_list, key=lambda s: int(osp.basename(s)[6:9]))
        img_path_list = sorted(img_path_list, key=lambda s: int(osp.basename(s)[1:4]))
        den_path_list = [p.replace('.png', '.h5').replace('imgs', 'dens') for p in img_path_list]

        for i in range(0, len(img_path_list), self.time_step):
            self.img_path_list.append(img_path_list[i: i + self.time_step])
            self.den_path_list.append(den_path_list[i: i + self.time_step])

    def load_files(self, item):
        imgs, dens = [], []
        img_path_list = self.img_path_list[item]
        den_path_list = self.den_path_list[item]

        for i in range(len(img_path_list)):
            img_path = img_path_list[i]
            den_path = den_path_list[i]

            img = Image.open(img_path).convert('RGB')

            try:
                with h5py.File(den_path, 'r') as hf:
                    den = np.asarray(hf['density'])
            except OSError:
                print('Failed to open file ', den_path)

            imgs.append(img)
            dens.append(den)

        return imgs, dens


# --- Loading with dot maps ---
# class UCSD(BaseLoader):
#     def __init__(self, data_dir, transforms, crop_size=400, scale=8, mode='train'):
#         super(UCSD, self).__init__(data_dir, transforms, crop_size, scale, mode)
#
#         self.time_step = scale
#
#         self.get_file_path()
#
#     def __getitem__(self, item):
#         imgs, dots, dens = self.load_files(item)
#
#         for i, [img, dot, den] in enumerate(zip(imgs, dots, dens)):
#             # imgs[i] = torch.from_numpy(img).float().unsqueeze(0)
#             imgs[i] = self.transforms(img)
#             dots[i] = torch.from_numpy(dot).float().unsqueeze(0)
#             dens[i] = torch.from_numpy(den).float().unsqueeze(0)
#
#         imgs = torch.stack(imgs)
#         dots = torch.stack(dots)
#         dens = torch.stack(dens)
#
#         if self.mode == 'train':
#             # random horizontal flip
#             if random.random() < 0.5:
#                 imgs = torch.flip(imgs, dims=[-1])
#                 dots = torch.flip(dots, dims=[-1])
#                 dens = torch.flip(dens, dims=[-1])
#
#             pts_list = []
#             # for i in range(self.scale):
#             #     dot = np.asarray(dots[i])
#             #     pts = np.array(np.where(dot > 0)).transpose()
#             #     permutation = [1, 0]
#             #     idx = np.empty_like(permutation)
#             #     idx[permutation] = np.arange(len(permutation))
#             #     pts[:] = pts[:, idx]
#             #     pts = torch.from_numpy(pts).float()
#             #     pts_list.append(pts)
#
#             return imgs, dots, dens, pts_list
#         elif self.mode == 'test':
#             return imgs, dots, dens
#
#     def __len__(self):
#         return int(len(self.img_path_list) / self.scale)
#
#     def get_file_path(self):
#         sub_data_dir = 'train_data' if self.mode == 'train' else 'test_data'
#         img_dir = osp.join(self.data_dir, sub_data_dir, 'imgs')
#
#         path_list = glob.glob(osp.join(img_dir, '*.png'))
#         path_list = sorted(path_list, key=lambda s: int(osp.basename(s)[6:9]))
#         path_list = sorted(path_list, key=lambda s: int(osp.basename(s)[1:4]))
#
#         for img_path in path_list:
#             dot_path = img_path.replace('.png', '.h5').replace('imgs', 'dots')
#             den_path = dot_path.replace('dots', 'dens')
#
#             self.img_path_list.append(img_path)
#             self.dot_path_list.append(dot_path)
#             self.den_path_list.append(den_path)
#
#     def load_files(self, item):
#         imgs, dots, dens = [], [], []
#
#         for i in range(self.scale):
#             index = int(item * self.scale + i)
#             img_path = self.img_path_list[index]
#             dot_path = self.dot_path_list[index]
#             den_path = self.den_path_list[index]
#
#             # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             img = Image.open(img_path).convert('RGB')
#
#             with h5py.File(dot_path, 'r') as hf:
#                 dot = np.asarray(hf['dot'])
#
#             if osp.isfile(den_path):
#                 with h5py.File(den_path, 'r') as hf:
#                     den = np.asarray(hf['density'])
#             else:
#                 den = dot
#
#             imgs.append(img)
#             dots.append(dot)
#             dens.append(den)
#
#         return imgs, dots, dens


# --- Loading UCSD processed by STDNet (TMM 2021) ---
# class UCSD(BaseLoader):
#     def __init__(self, data_dir, transforms, crop_size=400, scale=8, mode='train'):
#         super(UCSD, self).__init__(data_dir, transforms, crop_size, scale, mode)
#
#         self.time_step = scale
#
#         file_path_1 = osp.join(self.data_dir, 'UCSD_x_data.mat')
#         file_path_2 = osp.join(self.data_dir, 'UCSD_label_data.mat')
#
#         X_raw = loadmat(file_path_1)
#         Y_raw = loadmat(file_path_2)
#
#         ROI = Y_raw['roi'].reshape((158, 238, 1))
#
#         X = X_raw['X'][:, :] * ROI                  # [10, 200, 158, 238, 1]
#         Y = Y_raw['density_map'][:, :] * ROI        # [10, 200, 158, 238, 1]
#
#         del X_raw, Y_raw
#
#         if self.mode == 'train':
#             index = [3, 4, 5, 6]
#         else:
#             index = [0, 1, 2, 7, 8, 9]
#
#         X, Y = X[index], Y[index]
#
#         self.img_seqs = X.reshape([-1, self.time_step, 158, 238, 1])
#         self.den_seqs = Y.reshape([-1, self.time_step, 158, 238, 1])
#
#     def __getitem__(self, item):
#         imgs, dens = [], []
#
#         for img, den in zip(self.img_seqs[item], self.den_seqs[item]):
#             img = torch.from_numpy(img).float()             # [158, 238, 1]
#             den = torch.from_numpy(den).float()             # [158, 238, 1]
#             imgs.append(img)
#             dens.append(den)
#
#         imgs = torch.stack(imgs)                            # [time_step, 158, 238, 1]
#         dens = torch.stack(dens)                            # [time_step, 158, 238, 1]
#         imgs = imgs.permute(0, 3, 1, 2).contiguous()        # [time_step, 1, 158, 238]
#         dens = dens.permute(0, 3, 1, 2).contiguous()        # [time_step, 1, 158, 238]
#
#         if self.mode == 'train':
#             # Random horizontal flip
#             if random.random() < 0.5:
#                 imgs = torch.flip(imgs, dims=[-1])
#                 dens = torch.flip(dens, dims=[-1])
#
#             return imgs, dens
#         else:
#             return imgs, dens
#
#     def __len__(self):
#         return self.img_seqs.shape[0]
