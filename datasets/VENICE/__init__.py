import os
import cv2
import glob
import h5py
import torch
import random
import numpy as np
import os.path as osp
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter

from datasets import BaseLoader
from misc.utilities import vis_density, vis_dot_map


class Venice(BaseLoader):
    def __init__(self, data_dir, transforms, crop_size=400, scale=8, mode='train'):
        super(Venice, self).__init__(data_dir, transforms, crop_size, scale, mode)

        self.time_step = scale

        self.get_file_path()

    def __getitem__(self, item):
        imgs, dens = self.load_files(item)

        for i in range(len(imgs)):
            imgs[i] = self.transforms(imgs[i])
            dens[i] = torch.from_numpy(dens[i]).float().unsqueeze(0)

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
        return len(self.img_path_list)

    def get_file_path(self):
        sub_data_dir = 'train_data' if self.mode == 'train' else 'test_data'
        img_dir = osp.join(self.data_dir, sub_data_dir, 'imgs')

        img_path_list = glob.glob(osp.join(img_dir, '*.jpg'))
        img_path_list = sorted(img_path_list, key=lambda s: int(osp.basename(s)[7:11]))
        img_path_list = sorted(img_path_list, key=lambda s: int(osp.basename(s)[0:4]))
        den_path_list = [p.replace('.jpg', '.h5').replace('imgs', 'dens') for p in img_path_list]

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


def preprocess(src_data_root, dst_data_root, dst_shape=None,
               gen_img=True, gen_den=True, gen_dot=False,
               vis_den=False, vis_dot=False) -> None:
    print('Processed files will be saved in: ', dst_data_root)
    os.makedirs(dst_data_root, exist_ok=True)

    for mode in ['train', 'test']:
        print('-' * 10, f'Processing {mode} data', '-' * 50)

        src_data_dir = osp.join(src_data_root, mode + '_data')
        dst_data_dir = osp.join(dst_data_root, mode + '_data')

        dst_img_dir = osp.join(dst_data_dir, 'imgs')
        dst_den_dir = osp.join(dst_data_dir, 'dens')
        dst_dot_dir = osp.join(dst_data_dir, 'dots')
        if gen_img:
            os.makedirs(dst_img_dir, exist_ok=True)
        if gen_den:
            os.makedirs(dst_den_dir, exist_ok=True)
        if gen_dot:
            os.makedirs(dst_dot_dir, exist_ok=True)

        src_img_path_list = glob.glob(osp.join(src_data_dir, 'images', '*.jpg'))
        if len(src_img_path_list) == 0:
            print('Error: no images found in ', src_data_root)
            return
        else:
            src_img_path_list = [p for p in src_img_path_list if not osp.basename(p).endswith(').jpg')]
            src_img_path_list = sorted(src_img_path_list, key=lambda s: int(s.split('.')[0][-4:]))

        for i, src_img_path in enumerate(src_img_path_list):
            print(' [image: {:3d}/{:3d}] Path: {:s}'.format(
                i + 1, len(src_img_path_list), src_img_path))

            src_ann_path = src_img_path.replace('.jpg', '.mat').replace('images', 'ground-truth')
            save_name = osp.basename(src_img_path)[:-4]

            dst_img_path = osp.join(dst_img_dir, save_name + '.jpg')
            dst_den_path = osp.join(dst_den_dir, save_name + '.h5')
            dst_dot_path = osp.join(dst_dot_dir, save_name + '.h5')

            img = cv2.imread(src_img_path)
            gt = loadmat(src_ann_path)['annotation']
            roi = loadmat(src_ann_path)['roi']
            roi = np.array(roi, dtype=img.dtype)

            src_h, src_w, _ = img.shape
            if dst_shape:
                dst_h, dst_w = dst_shape
            else:
                dst_h, dst_w = src_h, src_w
            rate_h, rate_w = src_h / dst_h, src_w / dst_w

            # resize img
            img = cv2.bitwise_or(img, img, mask=roi)
            img = cv2.resize(img, (dst_w, dst_h))

            # generate dot map & density
            resized_roi = cv2.resize(roi, (dst_w, dst_h))
            gt_count = len(gt)
            dot_map = np.zeros((dst_h, dst_w))

            for point in gt:
                x = min(int(point[0] / rate_w), dst_w)
                y = min(int(point[1] / rate_h), dst_h)
                if resized_roi[y, x]:
                    dot_map[y, x] = 1

            density = gaussian_filter(dot_map, sigma=15)

            if gen_img:
                cv2.imwrite(dst_img_path, img)

            if gen_den:
                with h5py.File(dst_den_path, 'w') as hf:
                    hf['density'] = density
                    hf.close()

            if gen_dot:
                with h5py.File(dst_dot_path, 'w') as hf:
                    hf['dot'] = dot_map
                    hf.close()

            if vis_den:
                vis_density(img, density, save_path=dst_den_path.replace(
                    '.h5', '_den.jpg'))
            if vis_dot:
                vis_dot_map(img, dot_map, save_path=dst_dot_path.replace(
                    '.h5', '_dot.jpg'))

    return
