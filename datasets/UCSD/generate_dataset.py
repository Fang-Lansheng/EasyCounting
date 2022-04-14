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

from misc.utilities import vis_density, vis_dot_map


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for preprocess dataset UCSD')

    parser.add_argument('--data-root', type=str,
                        default='~/workspace/datasets/UCSD',
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
        dst_data_root = osp.join(project_path, 'processed_data/UCSD')
    else:
        dst_data_root = osp.join(osp.abspath(dst_data_root), 'UCSD')
    print('Processed files will be saved in: ', osp.abspath(dst_data_root))

    print('-' * 10, f'Processing data', '-' * 50)

    src_img_dir = osp.join(src_data_root, 'vidf')
    src_ann_dir = osp.join(src_data_root, 'vidf-cvpr')
    dst_train_data_root = osp.join(dst_data_root, 'train_data')
    dst_test_data_root = osp.join(dst_data_root, 'test_data')
    dst_train_img_dir = osp.join(dst_train_data_root, 'imgs')
    dst_train_den_dir = osp.join(dst_train_data_root, 'dens')
    dst_test_img_dir = osp.join(dst_test_data_root, 'imgs')
    dst_test_den_dir = osp.join(dst_test_data_root, 'dens')

    os.makedirs(dst_train_img_dir, exist_ok=True)
    os.makedirs(dst_train_den_dir, exist_ok=True)
    os.makedirs(dst_test_img_dir, exist_ok=True)
    os.makedirs(dst_test_den_dir, exist_ok=True)

    src_seq_list = glob.glob(osp.join(src_ann_dir, '*_count_roi_mainwalkway.mat'))
    src_seq_list = [osp.basename(x).strip('_count_roi_mainwalkway.mat') for x in src_seq_list]
    src_seq_list = sorted(src_seq_list, key=lambda s: int(s.split('_')[-1]))

    roi_file_path = osp.join(src_ann_dir, 'vidf1_33_roi_mainwalkway.mat')
    try:
        roi_file = loadmat(roi_file_path)
    except FileNotFoundError:
        print('No such file or directory: ', roi_file_path)
        return
    roi = roi_file['roi']['mask'][0][0]  # shape: (158, 238)

    for i, seq_name in enumerate(src_seq_list):
        src_img_path_list = glob.glob(osp.join(src_img_dir, seq_name + '.y', '*.png'))
        src_img_path_list = sorted(src_img_path_list, key=lambda s: int(osp.basename(s)[-7:-4]))

        src_cnt_path = osp.join(src_ann_dir, seq_name + '_count_roi_mainwalkway.mat')
        src_den_path = osp.join(src_ann_dir, seq_name + '_frame_full.mat')

        cnt_all = loadmat(src_cnt_path)
        den_all = loadmat(src_den_path)['frame'][0]

        for j, src_img_path in enumerate(src_img_path_list):
            print(' [Seq: {:2d}/{:2d}] [Image: {:3d}/{:3d}] Path: {:s}'.format(
                i + 1, len(src_seq_list), j + 1, len(src_img_path_list), src_img_path))

            img_name = osp.basename(src_img_path)
            gt_points = den_all[j][0][0]['loc']
            gt_count = gt_points.shape[0]

            img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
            dot_map = np.zeros(img.shape)

            src_h, src_w = img.shape
            if resize_shape is None:
                dst_h, dst_w = src_h, src_w
            else:
                dst_h, dst_w = resize_shape[0], resize_shape[1]
            rate_h, rate_w = src_h / dst_h, src_w / dst_w

            # resize img
            img = cv2.bitwise_or(img, img, mask=roi)
            img = cv2.resize(img, (dst_w, dst_h))

            for point in gt_points:
                x = min(int(point[0] / rate_w), dst_w - 1)
                y = min(int(point[1] / rate_h), dst_h - 1)

                dot_map[y, x] = 1

            density = gaussian_filter(dot_map, sigma=5) * roi

            scene, frame = img_name[9:12], img_name[-7:-4]
            save_name = 's{:s}_f{:s}'.format(scene, frame)
            if 3 <= int(scene) <= 6:
                dst_img_path = osp.join(dst_train_img_dir, save_name + '.png')
                dst_den_path = osp.join(dst_train_den_dir, save_name + '.h5')
            else:
                dst_img_path = osp.join(dst_test_img_dir, save_name + '.png')
                dst_den_path = osp.join(dst_test_den_dir, save_name + '.h5')

            cv2.imwrite(dst_img_path, img)

            # with h5py.File(dst_den_path, 'w') as hf:
            #     hf['density'] = density
            #     hf.close()

            vis_density(img, density,
                        save_path=dst_den_path.replace('.h5', '_den.jpg'),
                        show_img=True)

            # vis_dot_map(img, dot_map,
            #             save_path=dst_den_path.replace('.h5', '_dot.jpg'),
            #             show_img=False)

    print('\nDone!\n')


if __name__ == '__main__':
    main()
