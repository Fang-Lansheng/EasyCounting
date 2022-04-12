import os
import os.path as osp
import cv2
import glob
import numpy as np
import h5py
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

from misc.utilities import vis_den

src_data_root = '~/workspace/datasets/UCSD/'
src_img_dir = osp.join(src_data_root, 'vidf')
src_ann_dir = osp.join(src_data_root, 'vidf-cvpr')

dst_data_root = osp.join(osp.abspath('../..'), 'data/UCSD')
dst_train_data_root = osp.join(dst_data_root, 'train_data')
dst_test_data_root = osp.join(dst_data_root, 'test_data')
dst_train_img_dir = osp.join(dst_train_data_root, 'imgs')
dst_train_den_dir = osp.join(dst_train_data_root, 'dens')
dst_train_dot_dir = osp.join(dst_train_data_root, 'dots')
dst_test_img_dir = osp.join(dst_test_data_root, 'imgs')
dst_test_den_dir = osp.join(dst_test_data_root, 'dens')
dst_test_dot_dir = osp.join(dst_test_data_root, 'dots')
os.makedirs(dst_data_root, exist_ok=True)
os.makedirs(dst_train_data_root, exist_ok=True)
os.makedirs(dst_test_data_root, exist_ok=True)
os.makedirs(dst_train_img_dir, exist_ok=True)
os.makedirs(dst_train_den_dir, exist_ok=True)
os.makedirs(dst_train_dot_dir, exist_ok=True)
os.makedirs(dst_test_img_dir, exist_ok=True)
os.makedirs(dst_test_den_dir, exist_ok=True)
os.makedirs(dst_test_dot_dir, exist_ok=True)

print('Processed data saved in ', dst_data_root)

src_seq_name_list = glob.glob(osp.join(src_ann_dir, '*_count_roi_mainwalkway.mat'))
src_seq_name_list = [osp.basename(x).strip('_count_roi_mainwalkway.mat') for x in src_seq_name_list]
src_seq_name_list = sorted(src_seq_name_list, key=lambda s: int(s.split('_')[-1]))

roi_file = loadmat(osp.join(src_ann_dir, 'vidf1_33_roi_mainwalkway.mat'))
roi = roi_file['roi']['mask'][0][0]     # shape: (158, 238)

for seq_name in src_seq_name_list:
    src_img_path_list = glob.glob(osp.join(src_img_dir, seq_name + '.y', '*.png'))
    src_img_path_list = sorted(src_img_path_list, key=lambda s: int(osp.basename(s)[-7:-4]))

    src_cnt_path = osp.join(src_ann_dir, seq_name + '_count_roi_mainwalkway.mat')
    src_den_path = osp.join(src_ann_dir, seq_name + '_frame_full.mat')

    cnt_all = loadmat(src_cnt_path)
    den_all = loadmat(src_den_path)['frame'][0]

    for i, img_path in enumerate(src_img_path_list):
        img_name = osp.basename(img_path)
        gt_points = den_all[i][0][0]['loc']
        gt_count = gt_points.shape[0]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        den = np.zeros(img.shape)
        dot = np.zeros(img.shape)

        img = img * roi

        for point in gt_points:
            point2d = np.zeros(img.shape, dtype=np.float32)
            y, x, _ = np.floor(point + 0.5)
            x = int(x if x < img.shape[0] else img.shape[0] - 1)
            y = int(y if y < img.shape[1] else img.shape[1] - 1)

            point2d[x][y] = 1
            dot[x][y] = 1

            # den += gaussian_filter(point2d, sigma=3, mode='constant')
            den += gaussian_filter(point2d, sigma=5, mode='constant')

        den = den * roi
        dot = dot * roi

        den = den * np.sum(dot) / (np.sum(den) + 1e-6)

        scene, frame = img_name[9:12], img_name[-7:-4]
        save_name = 's{:s}_f{:s}'.format(scene, frame)
        if 3 <= int(scene) <= 6:
            dst_img_path = osp.join(dst_train_img_dir, save_name + '.png')
            dst_den_path = osp.join(dst_train_den_dir, save_name + '.h5')
            dst_dot_path = osp.join(dst_train_dot_dir, save_name + '.h5')
        else:
            dst_img_path = osp.join(dst_test_img_dir, save_name + '.png')
            dst_den_path = osp.join(dst_test_den_dir, save_name + '.h5')
            dst_dot_path = osp.join(dst_test_dot_dir, save_name + '.h5')

        # vis_den(img, den, save_path=dst_den_path.replace('.h5', '.jpg'))
        cv2.imwrite(dst_img_path, img)
        with h5py.File(dst_den_path, 'w') as hf:
            hf['density'] = den
        # with h5py.File(dst_dot_path, 'w') as hf:
        #     hf['dot'] = dot

print('\nDone!\n')
