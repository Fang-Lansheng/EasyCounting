import os
import cv2
import glob
import h5py
import pathlib
import argparse
import numpy as np
import os.path as osp
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat

from misc.utilities import generate_density, vis_density, vis_dot_map


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for preprocess dataset ShanghaiTech Part_A')

    parser.add_argument('--data-root', type=str,
                        default='~/workspace/datasets/SHHA',
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
        dst_data_root = osp.join(project_path, 'processed_data/SHHA')
    else:
        dst_data_root = osp.join(osp.abspath(dst_data_root), 'SHHA')
    print('Processed files will be saved in: ', osp.abspath(dst_data_root))

    for mode in ['train', 'test']:
        print('-' * 10, f'Processing {mode} data', '-' * 50)

        src_data_dir = osp.join(src_data_root, mode + '_data')
        dst_data_dir = osp.join(dst_data_root, mode + '_data')

        dst_img_dir = osp.join(dst_data_dir, 'imgs')
        dst_den_dir = osp.join(dst_data_dir, 'dens')
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_den_dir, exist_ok=True)

        src_img_path_list = glob.glob(osp.join(src_data_dir, 'images', '*.jpg'))
        if len(src_img_path_list) == 0:
            print('Error: no images found in ', src_data_root)
            return
        else:
            src_img_path_list = sorted(src_img_path_list,
                                       key=lambda s: int(osp.basename(s)[4:-4]))

        for i, src_img_path in enumerate(src_img_path_list):
            print(' [image: {:3d}/{:3d}] Path: {:s}'.format(
                i + 1, len(src_img_path_list), src_img_path))

            save_name = osp.basename(src_img_path)[:-4]
            src_ann_path = osp.join(src_data_dir, 'ground_truth',
                                    'GT_' + save_name + '.mat')

            dst_img_path = osp.join(dst_img_dir, save_name + '.jpg')
            dst_den_path = osp.join(dst_den_dir, save_name + '.h5')

            img = cv2.imread(src_img_path)
            gt = loadmat(src_ann_path)['image_info'][0, 0][0, 0][0]

            src_h, src_w, _ = img.shape
            if resize_shape is None:
                dst_h, dst_w = src_h, src_w
            else:
                dst_h, dst_w = resize_shape[0], resize_shape[1]
            rate_h, rate_w = src_h / dst_h, src_w / dst_w

            # resize img
            img = cv2.resize(img, (dst_w, dst_h))

            # generate dot map & density
            gt_count = len(gt)
            dot_map = np.zeros((dst_h, dst_w))

            for point in gt:
                x = min(int(point[0] / rate_w), dst_w - 1)
                y = min(int(point[1] / rate_h), dst_h - 1)
                dot_map[y, x] = 1

            # generate density map with adaptive kernel size
            density = generate_density(dot_map)
            # generate density map with fixed kernel size
            # density = gaussian_filter(dot_map, sigma=5)

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
