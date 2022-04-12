import os
import glob
import cv2
import numpy as np
import os.path as osp
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat

from misc.utilities import vis_den, vis_dot


src_data_root = '/Users/jifanzhang/workspace/Dev/Datasets/Venice'
dst_data_root = osp.join(osp.abspath('../..'), 'data/VENICE')
print('Processed files will be saved in: ', dst_data_root)

for mode in ['train', 'test']:
    print('-' * 10, f'Processing {mode} data', '-' * 50)

    src_data_dir = osp.join(src_data_root, mode + '_data')
    dst_data_dir = osp.join(dst_data_root, mode + '_data')

    dst_img_dir = osp.join(dst_data_dir, 'imgs')
    dst_den_dir = osp.join(dst_data_dir, 'dens')
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_den_dir, exist_ok=True)

    src_img_path_list = glob.glob(osp.join(src_data_dir, 'images', '*.jpg'))
    src_img_path_list = [p for p in src_img_path_list if not osp.basename(p).endswith(').jpg')]
    src_img_path_list = sorted(src_img_path_list, key=lambda s: int(s.split('.')[0][-4:]))

    for i, src_img_path in enumerate(src_img_path_list):
        # print(' [image: {:3d}/{:3d}] Path: {:s}'.format(
        #     i + 1, len(src_img_path_list), src_img_path))

        src_ann_path = src_img_path.replace('.jpg', '.mat').replace('images', 'ground-truth')
        save_name = osp.basename(src_img_path)[:-4]

        dst_img_path = osp.join(dst_img_dir, save_name + '.jpg')
        dst_den_path = osp.join(dst_den_dir, save_name + '.h5')

        img = cv2.imread(src_img_path)
        gt = loadmat(src_ann_path)['annotation']
        roi = loadmat(src_ann_path)['roi']
        roi = np.array(roi, dtype=img.dtype)

        src_h, src_w, _ = img.shape
        # dst_h, dst_w = 360, 640
        dst_h, dst_w = src_h, src_w
        rate_h, rate_w = src_h / dst_h, src_w / dst_w

        # if i == 12:
        #     print('')

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

        cv2.imwrite(dst_img_path, img)

        with h5py.File(dst_den_path, 'w') as hf:
            hf['density'] = density
            hf.close()

        # vis_den(img, density, save_path=dst_den_path.replace('.h5', '_den.jpg'))
        # vis_dot(img, dot_map, save_path=dst_den_path.replace('.h5', '_dot.jpg'))
        #
        # print(' [{:3d}/{:3d}] Name: {:s} | GT: {:4d} | DOT: {:6.2f} | DEN: {:6.2f}'.format(
        #     i + 1, len(src_img_path_list), osp.basename(src_img_path), gt_count, np.sum(dot_map), np.sum(density)))

print('\nDone!\n')
