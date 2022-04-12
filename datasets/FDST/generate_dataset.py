import os
import glob
import cv2
import h5py
import json
import numpy as np
import os.path as osp
from scipy.ndimage.filters import gaussian_filter

src_data_root = '~/workspace/datasets/FDST'
dst_data_root = osp.join(osp.abspath('../..'), 'data/FDST')
print('Processed files will be saved in: ', dst_data_root)

# for mode in ['train', 'test']:
for mode in ['test', 'train']:
    print('-' * 10, f'Processing {mode} data', '-' * 50)

    src_data_dir = osp.join(src_data_root, mode + '_data')
    dst_data_dir = osp.join(dst_data_root, mode + '_data')

    dst_img_dir = osp.join(dst_data_dir, 'imgs')
    dst_dot_dir = osp.join(dst_data_dir, 'dots')
    dst_den_dir = osp.join(dst_data_dir, 'dens')
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_dot_dir, exist_ok=True)
    os.makedirs(dst_den_dir, exist_ok=True)

    scene_list = [p for p in os.listdir(src_data_dir) if not p.startswith('.')]
    assert len(scene_list) % 10 == 0
    scene_list = sorted(scene_list, key=lambda s: int(s))

    for i, scene_name in enumerate(scene_list):
        src_img_path_list = glob.glob(osp.join(src_data_dir, scene_name, '*.jpg'))
        src_img_path_list = [p for p in src_img_path_list if not osp.basename(p).endswith(').jpg')]
        assert len(src_img_path_list) == 150
        src_img_path_list = sorted(src_img_path_list, key=lambda s: int(s.split('.')[0][-3:]))

        for j, src_img_path in enumerate(src_img_path_list):
            print(' [scene: {:2d}/{:2d}] [image: {:3d}/{:3d}] Path: {:s}'.format(
                i + 1, len(scene_list), j + 1, len(src_img_path_list), src_img_path))

            frame_name = osp.basename(src_img_path)[:3]
            src_ann_path = src_img_path.replace('.jpg', '.json')

            save_name = 's{:03d}_f{:s}'.format(int(scene_name), frame_name)

            dst_img_path = osp.join(dst_img_dir, save_name + '.jpg')
            dst_dot_path = osp.join(dst_dot_dir, save_name + '.h5')
            dst_den_path = osp.join(dst_den_dir, save_name + '.h5')

            img = cv2.imread(src_img_path)
            with open(src_ann_path, 'r') as f:
                gt = json.load(f)
            gt = list(gt.values())[0]['regions']

            src_h, src_w, _ = img.shape
            dst_h, dst_w = 360, 640
            rate_h, rate_w = src_h / dst_h, src_w / dst_w

            # resize img
            img = cv2.resize(img, (dst_w, dst_h))

            # generate dot map & density
            gt_count = len(gt)
            dot_map = np.zeros((dst_h, dst_w))

            for ann in gt:
                src_rect_x_top_left = ann['shape_attributes']['x']
                src_rect_y_top_left = ann['shape_attributes']['y']
                src_rect_width = ann['shape_attributes']['width']
                src_rect_height = ann['shape_attributes']['height']
                src_x = src_rect_x_top_left + 0.5 * src_rect_width
                src_y = src_rect_y_top_left + 0.5 * src_rect_height
                x = min(int(src_x / rate_w), dst_w)
                y = min(int(src_y / rate_h), dst_h)
                dot_map[y, x] = 1

            density = gaussian_filter(dot_map, sigma=5)

            cv2.imwrite(dst_img_path, img)
            # with h5py.File(dst_dot_path, 'w') as hf:
            #     hf['dot'] = dot_map
            #     hf.close()
            with h5py.File(dst_den_path, 'w') as hf:
                hf['density'] = density
                hf.close()

