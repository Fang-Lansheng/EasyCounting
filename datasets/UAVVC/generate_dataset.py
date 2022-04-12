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

src_data_root = '/Users/jifanzhang/workspace/Dev/Datasets/UAV_CAR_CC'
dst_data_root = osp.join(osp.abspath('../..'), 'data/UAVVC')
print('Processed files will be saved in: ', dst_data_root)
os.makedirs(dst_data_root, exist_ok=True)

for mode in ['train', 'test']:
    print('-' * 10, f'Processing {mode} data', '-' * 50)
    src_img_list = glob.glob(osp.join(src_data_root, mode, 'image', '*.jpg'))

    dst_data_dir = osp.join(dst_data_root, mode + '_data')
    dst_img_dir = osp.join(dst_data_dir, 'imgs')
    dst_den_dir = osp.join(dst_data_dir, 'dens')
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_den_dir, exist_ok=True)

    for i, src_img_path in enumerate(src_img_list):
        # source files paths
        src_img_name = osp.basename(src_img_path)    # ignore '\n' at the end of line
        frame = int(src_img_name.split('_')[0])
        sequence = int(src_img_name[-10:-4])
        src_ann_path = src_img_path.replace('image', 'anno').replace('.jpg', '.mat')

        print(' [image: {:3d}/{:3d}] Path: {:s}'.format(i + 1, len(src_img_list), src_img_path))

        # open source files
        src_img = cv2.imread(src_img_path)
        src_ann = loadmat(src_ann_path)['anno']

        # data processing
        src_h, src_w, _ = src_img.shape
        dst_h, dst_w = src_h, src_w
        rate_h, rate_w = src_h / dst_h, src_w / dst_w

        dst_img = cv2.resize(src_img, (dst_w, dst_h))
        dst_dot = np.zeros((dst_h, dst_w))

        for point in src_ann:
            src_x, src_y = (point[0] + point[2]) / 2, (point[1] + point[3]) / 2
            if 0 <= src_x < src_w and 0 <= src_y < src_h:
                dst_x = min(int(src_x / rate_w), dst_w)
                dst_y = min(int(src_y / rate_h), dst_h)
                dst_dot[dst_y, dst_x] = 1

        dst_den = gaussian_filter(dst_dot, sigma=10)

        # destination files paths
        dst_img_name = 'f{:02d}_s{:04d}.jpg'.format(frame, sequence)
        dst_img_path = osp.join(dst_img_dir, dst_img_name)
        dst_den_path = osp.join(dst_den_dir, dst_img_name.replace('.jpg', '.h5'))

        # save destination files
        cv2.imwrite(dst_img_path, dst_img)

        with h5py.File(dst_den_path, 'w') as hf:
            hf['density'] = dst_den

        # dst_vis_dir = dst_img_dir.replace('imgs', 'vis')
        # os.makedirs(dst_vis_dir, exist_ok=True)
        # vis_den(dst_img, dst_den, save_path=dst_img_path.replace('imgs', 'vis'))

print('\nDone!\n')
