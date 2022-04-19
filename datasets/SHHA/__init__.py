import os
import glob
from datasets import BaseLoader


class SHHA(BaseLoader):
    def __init__(self, data_dir, transforms, crop_size=400, scale=8, mode='train'):
        super(SHHA, self).__init__(data_dir, transforms, crop_size, scale, mode)
        self.get_file_path()

    def get_file_path(self):
        sub_data_dir = 'train_data' if self.mode == 'train' else 'test_data'
        img_dir_path = os.path.join(self.data_dir, sub_data_dir, 'images')

        path_list = glob.glob(os.path.join(img_dir_path, '*.jpg'))
        path_list = sorted(path_list,
                           key=lambda s: int(s.split('IMG_')[1][:-4]))

        for img_path in path_list:
            dot_path = img_path.replace('.jpg', '.h5').replace('images', 'dot_map')
            den_path = dot_path.replace('dot_map', 'density')

            self.img_path_list.append(img_path)
            self.dot_path_list.append(dot_path)
            self.den_path_list.append(den_path)
