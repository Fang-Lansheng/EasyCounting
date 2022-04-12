import os
import glob
import cv2
import h5py
import json
import numpy as np
import pandas as pd
import os.path as osp
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat
import matplotlib.pyplot as plt

from misc.utilities import vis_den


src_data_root = '~/workspace/datasets/TRANCOS_v3'
dst_data_root = osp.join(osp.abspath('../..'), 'data/TRANCOS')
print('Processed files will be saved in: ', dst_data_root)
os.makedirs(dst_data_root, exist_ok=True)

print('-' * 10, f'Processing data', '-' * 50)

for mode in ['train', 'test']:
    if mode == 'train':
        with open(osp.join(src_data_root, 'image_sets', 'trainval.txt')) as f:
            src_img_list = f.readlines()
    else:
        with open(osp.join(src_data_root, 'image_sets', 'test.txt')) as f:
            src_img_list = f.readlines()

    dst_data_dir = osp.join(dst_data_root, mode + '_data')
    dst_img_dir = osp.join(dst_data_dir, 'imgs')
    dst_den_dir = osp.join(dst_data_dir, 'dens')
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_den_dir, exist_ok=True)

    for i, src_img_name in enumerate(src_img_list):
        # source files paths
        src_img_name = src_img_name[:-1]    # ignore '\n' at the end of line
        src_img_path = osp.join(src_data_root, 'images', src_img_name)
        src_mask_path = osp.join(src_img_path.replace('.jpg', 'mask.mat'))
        src_ann_path = osp.join(src_img_path.replace('.jpg', '.txt'))
        src_dots_path = osp.join(src_img_path.replace('.jpg', 'dots.png'))

        print(' [image: {:3d}/{:3d}] Path: {:s}'.format(i + 1, len(src_img_list), src_img_path))

        # open source files
        src_img = cv2.imread(src_img_path)
        src_roi = loadmat(src_mask_path)['BW']
        src_dot = cv2.imread(src_dots_path)
        src_ann = pd.read_table(src_ann_path, header=None).values

        # data processing
        src_h, src_w, _ = src_img.shape
        dst_h, dst_w = src_h, src_w
        rate_h, rate_w = src_h / dst_h, src_w / dst_w

        src_roi = np.array(src_roi, dtype=src_img.dtype)
        dst_img = cv2.bitwise_or(src_img, src_img, mask=src_roi)
        dst_img = cv2.resize(dst_img, (dst_w, dst_h))

        # src_dot = np.where(src_dot > 0)[0:2]
        # src_dot = np.asarray([src_dot[1], src_dot[0]]).transpose()
        # src_dot = sorted(src_dot, key=lambda p: int(p[0]))

        dst_dot = np.zeros((dst_h, dst_w))
        for point in src_ann:
            src_x, src_y = point
            if 0 <= src_x < src_w and 0 <= src_y < src_h and src_roi[src_y, src_x] > 0:
                dst_x = min(int(src_x / rate_w), dst_w)
                dst_y = min(int(src_y / rate_h), dst_h)
                dst_dot[dst_y, dst_x] = 1

        dst_den = gaussian_filter(dst_dot, sigma=10)

        # destination files paths
        dst_img_path = osp.join(dst_img_dir, src_img_name)
        dst_den_path = osp.join(dst_den_dir, src_img_name.replace('.jpg', '.h5'))

        # save destination files
        cv2.imwrite(dst_img_path, dst_img)

        with h5py.File(dst_den_path, 'w') as hf:
            hf['density'] = dst_den

        vis_den(dst_img, dst_den, save_path=dst_den_path.replace('.h5', '.jpg'))

print('\nDone!\n')
