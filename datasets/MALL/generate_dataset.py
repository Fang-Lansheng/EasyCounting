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
        description='Script for preprocess dataset Mall')

    parser.add_argument('--data-root', type=str,
                        default='~/workspace/datasets/MALL',
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
        dst_data_root = osp.join(project_path, 'processed_data/MALL')
    else:
        dst_data_root = osp.join(osp.abspath(dst_data_root), 'MALL')
    print('Processed files will be saved in: ', osp.abspath(dst_data_root))

    src_img_path_list = glob.glob(osp.join(src_data_root, 'frames', '*.jpg'))
    if len(src_img_path_list) == 0:
        print('Error: no images found in ', src_data_root)
        return
    else:
        src_img_path_list = sorted(src_img_path_list,
                                   key=lambda s: int(s.split('.')[0][-4:]))

    gt = loadmat(osp.join(src_data_root, 'mall_gt.mat'))
    feat = loadmat(osp.join(src_data_root, 'mall_feat.mat'))
    roi = loadmat(osp.join(src_data_root, 'perspective_roi.mat'))

    gt_frame = gt['frame']
    gt_count = gt['count']
    roi = roi['roi'][0][0][0]

    for i, src_img_path in enumerate(src_img_path_list):
        print(' [image: {:3d}/{:3d}] Path: {:s}'.format(
            i + 1, len(src_img_path_list), src_img_path))
        mode = 'train' if i < 800 else 'test'

        dst_data_dir = osp.join(dst_data_root, mode + '_data')
        dst_img_dir = osp.join(dst_data_dir, 'imgs')
        dst_den_dir = osp.join(dst_data_dir, 'dens')
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_den_dir, exist_ok=True)

        save_name = osp.basename(src_img_path)[:-4]

        dst_img_path = osp.join(dst_img_dir, save_name + '.jpg')
        dst_den_path = osp.join(dst_den_dir, save_name + '.h5')

        img = cv2.imread(src_img_path)

        count = gt_count[i][0]
        points = gt_frame[0][i][0][0][0]
        assert count == points.shape[0]

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
        dot_map = np.zeros((dst_h, dst_w))

        for point in points:
            x = min(int(point[0] / rate_w), dst_w)
            y = min(int(point[1] / rate_h), dst_h)
            dot_map[y, x] = 1

        density = gaussian_filter(dot_map, sigma=5)

        cv2.imwrite(dst_img_path, img)

        with h5py.File(dst_den_path, 'w') as hf:
            hf['density'] = density
            hf.close()

        # vis_density(img, density,
        #             save_path=dst_den_path.replace('.h5', '_den.jpg'),
        #             show_img=True)

    print('\nDone!\n')


if __name__ == '__main__':
    main()
