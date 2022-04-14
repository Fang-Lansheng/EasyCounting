import os
import cv2
import glob
import h5py
import pathlib
import argparse
import numpy as np
import pandas as pd
import os.path as osp
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat

from misc.utilities import vis_density, vis_dot_map


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for preprocess dataset Trancos')

    parser.add_argument('--data-root', type=str,
                        default='~/workspace/datasets/Trancos',
                        help='path to the raw dataset')
    parser.add_argument('--destination', type=str,
                        default=None,
                        help='path to the processed data')
    parser.add_argument('--resize-shape', type=int, nargs='+',
                        default=None,
                        help='path to the processed data')

    return parser.parse_args()


def main():
    args = parse_args()
    src_data_root = args.data_root
    dst_data_root = args.destination
    resize_shape = args.resize_shape

    if dst_data_root is None:
        project_path = pathlib.Path(__file__).parent.parent.parent
        dst_data_root = osp.join(project_path, 'processed_data/TRANCOS')
    else:
        dst_data_root = osp.join(osp.abspath(dst_data_root), 'TRANCOS')
    print('Processed files will be saved in: ', osp.abspath(dst_data_root))

    for mode in ['train', 'test']:
        print('-' * 10, f'Processing {mode} data', '-' * 50)

        dst_data_dir = osp.join(dst_data_root, mode + '_data')
        dst_img_dir = osp.join(dst_data_dir, 'imgs')
        dst_den_dir = osp.join(dst_data_dir, 'dens')
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_den_dir, exist_ok=True)

        src_img_list = []
        if mode == 'train':
            fp = osp.join(src_data_root, 'image_sets', 'trainval.txt')
        else:
            fp = osp.join(src_data_root, 'image_sets', 'test.txt')

        with open(fp) as f:
            for line in f.readlines():
                src_img_list.append(line[:-1])

        if len(src_img_list) == 0:
            print('Error: no images found in ', src_data_root)
            return

        for i, src_img_name in enumerate(src_img_list):
            src_img_path = osp.join(src_data_root, 'images', src_img_name)
            src_mask_path = osp.join(src_img_path.replace('.jpg', 'mask.mat'))
            src_ann_path = osp.join(src_img_path.replace('.jpg', '.txt'))

            save_name = src_img_name[:-4]
            dst_img_path = osp.join(dst_img_dir, save_name + '.jpg')
            dst_den_path = osp.join(dst_den_dir, save_name + '.h5')

            print(' [image: {:3d}/{:3d}] Path: {:s}'.format(
                i + 1, len(src_img_list), src_img_path))

            img = cv2.imread(src_img_path)
            roi = loadmat(src_mask_path)['BW']
            gt = pd.read_table(src_ann_path, header=None).values

            src_h, src_w, _ = img.shape
            if resize_shape is None:
                dst_h, dst_w = src_h, src_w
            else:
                dst_h, dst_w = resize_shape[0], resize_shape[1]
            rate_h, rate_w = src_h / dst_h, src_w / dst_w

            # resize img
            roi = np.array(roi, dtype=img.dtype)
            img = cv2.bitwise_or(img, img, mask=roi)
            img = cv2.resize(img, (dst_w, dst_h))

            # generate dot map & density
            resized_roi = cv2.resize(roi, (dst_w, dst_h))
            gt_count = len(gt)
            dot_map = np.zeros((dst_h, dst_w))

            for point in gt:
                x = min(int(point[0] / rate_w), dst_w - 1)
                y = min(int(point[1] / rate_h), dst_h - 1)
                if resized_roi[y, x]:
                    dot_map[y, x] = 1

            density = gaussian_filter(dot_map, sigma=10)

            cv2.imwrite(dst_img_path, img)

            with h5py.File(dst_den_path, 'w') as hf:
                hf['density'] = density
                hf.close()

            # vis_density(img, density,
            #             save_path=dst_den_path.replace('.h5', '_den.jpg'),
            #             show_img=True)

            # vis_dot_map(img, dot_map,
            #             save_path=dst_den_path.replace('.h5', '_dot.jpg'),
            #             show_img=False)

    print('\nDone!\n')


if __name__ == '__main__':
    main()
