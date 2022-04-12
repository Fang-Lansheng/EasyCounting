# -*- coding: <encoding name> -*-
"""
utils
"""
import os
import time
import math
import glob
import h5py
import torch
import random
import shutil
import logging
import argparse
import importlib
import scipy.spatial

import os.path as osp
import numpy as np
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from mmcv.utils import Config
from scipy.ndimage.filters import gaussian_filter


################################################################################
# change the learning rate according to epoch.
################################################################################
def adjust_learning_rate(optimizer, epoch):
    if (epoch + 1) % 100 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5


################################################################################
# set the random seed.
################################################################################
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


################################################################################
# dataloader collate function
################################################################################
def train_collate(batch):
    transposed_batch = list(zip(*batch))

    if len(transposed_batch) == 4:
        img = torch.stack(transposed_batch[0], 0)
        dot = torch.stack(transposed_batch[1], 0)
        den = torch.stack(transposed_batch[2], 0)
        pts = transposed_batch[3]

        return img, dot, den, pts

    elif len(transposed_batch) == 2:
        img = torch.stack(transposed_batch[0], 0)
        cnt = torch.stack(transposed_batch[1], 0)

        return img, cnt

    elif len(transposed_batch) == 8:
        img_1 = torch.stack(transposed_batch[0], 0)
        dot_1 = torch.stack(transposed_batch[1], 0)
        den_1 = torch.stack(transposed_batch[2], 0)
        pts_1 = transposed_batch[3]

        img_2 = torch.stack(transposed_batch[4], 0)
        dot_2 = torch.stack(transposed_batch[5], 0)
        den_2 = torch.stack(transposed_batch[6], 0)
        pts_2 = transposed_batch[7]

        return img_1, dot_1, den_1, pts_1, \
               img_2, dot_2, den_2, pts_2


################################################################################
# prepare data loader
################################################################################
def get_dataloader(cfg, mode='train'):
    scale = cfg.dataset.scale
    crop_size = cfg.dataset.crop_size
    mean = cfg.dataset.img_norm_cfg.mean
    std = cfg.dataset.img_norm_cfg.std

    data_root = cfg.dataset.data_root
    batch_size = cfg.dataset.batch_size
    num_workers = cfg.runner.num_workers

    data_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    if cfg.dataset.name.upper() == 'QNRF':
        from datasets.QNRF import QNRF as CCDataset
    elif cfg.dataset.name.upper() == 'NWPU':
        from datasets.SHHA import SHHA as CCDataset
    elif cfg.dataset.name.upper() == 'SHHA':
        from datasets.SHHA import SHHA as CCDataset
    elif cfg.dataset.name.upper() == 'SHHB':
        from datasets.SHHB import SHHB as CCDataset
    elif cfg.dataset.name.upper() == 'UCSD':
        from datasets.UCSD import UCSD as CCDataset
    elif cfg.dataset.name.upper() == 'FDST':
        from datasets.FDST import FDST as CCDataset
    elif cfg.dataset.name.upper() == 'VENICE':
        from datasets.VENICE import Venice as CCDataset
    elif cfg.dataset.name.upper() == 'MALL':
        from datasets.MALL import MALL as CCDataset
    elif cfg.dataset.name.upper() == 'TRANCOS':
        from datasets.TRANCOS import Trancos as CCDataset
    elif cfg.dataset.name.upper() == 'UAVVC':
        from datasets.UAVVC import UAVVC as CCDataset
    elif cfg.dataset.name.upper() == 'CARPK':
        from datasets.CARPK import CARPK as CCDataset
    else:
        raise NotImplementedError

    val_dataset = CCDataset(data_root, data_transform,
                            scale=scale, mode='test')
    val_loader = DataLoader(val_dataset, batch_size=1,
                            num_workers=num_workers, pin_memory=True)

    if mode == 'train':
        train_dataset = CCDataset(data_root, data_transform, scale=scale,
                                  crop_size=crop_size, mode='train')
        train_loader = DataLoader(train_dataset, collate_fn=train_collate,
                                  batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

        return [train_loader, val_loader]

    return val_loader


################################################################################
# prepare the model trainer
################################################################################
def get_trainer(cfg, data_loaders):
    if cfg.model.name == 'CSRNet':
        from models.CSRNet.trainer import Trainer
    elif cfg.model.name == 'PSNet':
        from models.PSNet.trainer import Trainer
    elif cfg.model.name == 'DM_Count':
        from models.DM_Count.trainer import Trainer
    elif cfg.model.name == 'STDNet':
        from models.STDNet.trainer import Trainer
    else:
        return NotImplementedError

    trainer = Trainer(cfg, data_loaders)

    return trainer


################################################################################
# prepare the model trainer
################################################################################
def get_tester(ckpt, cfg, val_loader, vis_options):
    if cfg.model.name == 'CSRNet':
        from models.CSRNet.tester import Tester
    elif cfg.model.name == 'DM_Count':
        from models.DM_Count.tester import Tester
    elif cfg.model.name == 'PSNet':
        from models.PSNet.tester import Tester
    elif 'STDNet' in cfg.model.name:
        from models.STDNet.tester import Tester
    else:
        return NotImplementedError

    tester = Tester(ckpt, cfg, val_loader, vis_options)

    return tester


################################################################################
# get logger and copy current environment
################################################################################
def prepare(cfg, mode='train'):
    if mode == 'train':
        if not osp.isfile(cfg.runner.resume):
            current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

            if cfg.runner.ckpt_dir == '':
                cfg.runner.ckpt_dir = osp.join(
                    cfg.runner.base_dir, 'Experiments',
                    cfg.model.name + '_' + cfg.dataset.name.upper() + '_' + current_time)
                os.makedirs('Experiments', exist_ok=True)
                os.makedirs(cfg.runner.ckpt_dir, exist_ok=True)

            copy_cur_env(cfg.runner.base_dir, cfg.runner.ckpt_dir + '/code')
        else:
            if cfg.runner.ckpt_dir == '':
                cfg.runner.ckpt_dir = osp.abspath(osp.dirname(cfg.runner.resume))

    log_file_name = 'train.log' if mode == 'train' else 'test.log'
    log_file_path = osp.join(cfg.runner.ckpt_dir, log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger_formatter = logging.Formatter(
        fmt='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    if mode == 'train':
        file_handler = logging.FileHandler(log_file_path, mode='a')
    else:
        file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(logger_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logger_formatter)
    logger.addHandler(console_handler)

    if mode == 'train':
        if not osp.isfile(cfg.runner.resume):
            logging.info('{} {}'.format('> TRAINING CONFIG ', '-' * 80))
            logging.info('\n{}'.format(cfg.pretty_text))
            logging.info('{} {}'.format('> START TRAINING  ', '-' * 80))
    else:
        logging.info('{} {}'.format('> TESTING CONFIG  ', '-' * 80))
        logging.info('\n{}'.format(cfg.pretty_text))
        logging.info('{} {}'.format('> START TESTING   ', '-' * 80))


################################################################################
# resize the tensor
################################################################################
def resize(input, size, flag=False):
    scale_factor_height = input.shape[-2] / size[-2]
    scale_factor_width = input.shape[-1] / size[-1]
    input = F.interpolate(input=input, size=size,
                          mode='bilinear', align_corners=True)
    if flag:
        input *= scale_factor_height * scale_factor_width
    return input


################################################################################
# save code
################################################################################
def copy_cur_env(work_dir, dst_dir, exception=None):
    if exception is None:
        exception = ['.git',  '.idea', '.backup', '__pycache__',
                     'data', 'logs', 'Experiments']

    if not osp.exists(dst_dir):
        os.mkdir(dst_dir)
    else:
        raise IOError("Dir \'" + dst_dir + "\' already exists!")

    for filename in os.listdir(work_dir):
        file = osp.join(work_dir, filename)
        dst_file = osp.join(dst_dir, filename)

        if osp.isdir(file) and filename not in exception:
            shutil.copytree(file, dst_file)
        elif osp.isfile(file):
            shutil.copyfile(file, dst_file)


################################################################################
# plot the density map.
################################################################################
def plot_density(dm, dm_dir, img_name, if_count=False):
    # type: (np.ndarray, str, str, bool) -> None
    assert osp.isdir(dm_dir)

    dm = dm[0, 0, :, :]
    count = np.sum(dm)

    dm = dm / np.max(dm + 1e-20)

    dm_frame = plt.gca()
    plt.imshow(dm, 'jet')

    dm_frame.axes.get_yaxis().set_visible(False)
    dm_frame.axes.get_xaxis().set_visible(False)
    dm_frame.spines['top'].set_visible(False)
    dm_frame.spines['bottom'].set_visible(False)
    dm_frame.spines['left'].set_visible(False)
    dm_frame.spines['right'].set_visible(False)

    if if_count:
        dm_file_name = img_name.split('.')[0] + '_cnt_%.2f.jpg' % count
    else:
        dm_file_name = img_name.split('.')[0]
    plt.savefig(osp.join(dm_dir, dm_file_name),
                bbox_inches='tight', pad_inches=0, dpi=150)

    plt.close()


################################################################################
# concat the raw image and the warped image
################################################################################
def plot_concat(img_raw, img_warped, save_dir, img_name):
    height, width = img_raw.height, img_raw.width

    dst = Image.new('RGB', (width * 2, height))
    dst.paste(img_raw, (0, 0))
    dst.paste(img_warped, (width, 0))

    dst.save(osp.join(save_dir, img_name))


################################################################################
# save density map as .h5 file
################################################################################
def save_h5(gt_warped, save_dir, img_name, name='dot'):
    # type: (np.ndarray, str, str, str) -> None
    gt_warped = gt_warped[0, 0, :, :]

    save_name = img_name.replace('.jpg', '.h5')
    save_path = osp.join(save_dir, save_name)

    with h5py.File(save_path, 'w') as hf:
        hf[name] = gt_warped


################################################################################
# draw grid on image
################################################################################
def draw_grid(x, grid_size=0, grid_interval=15, grid_color=1):
    # type: (torch.Tensor, int, int, int) -> torch.Tensor
    b, c, h, w = x.shape

    grid_color = torch.tensor([grid_color])

    if grid_size:
        dx = int(w / grid_size)
        dy = int(h / grid_size)

        x[:, :, ::dy, :] = grid_color
        x[:, :, :, ::dx] = grid_color
    if grid_interval:
        x[:, :, ::grid_interval, :] = grid_color
        x[:, :, :, ::grid_interval] = grid_color
    return x


################################################################################
# draw grid on image
################################################################################
def generate_density(dot_map):
    shape = dot_map.shape
    density = np.zeros(shape, dtype=np.float32)

    gt_count = np.count_nonzero(dot_map)

    if gt_count == 0:
        return density

    # points = np.array(zip(np.nonzero(dot_map)[1],
    #                       np.nonzero(dot_map)[0]))

    points = np.array(np.where(dot_map > 0), dtype=np.float64).transpose()

    tree = scipy.spatial.KDTree(points.copy(), leafsize=2048)
    distances, locations = tree.query(points, k=4)

    for i, pt in enumerate(points):
        point2d = np.zeros(shape, dtype=np.float32)
        point2d[int(pt[0]), int(pt[1])] = 1.

        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(shape)) / 2. / 2.

        density += gaussian_filter(point2d, sigma, mode='constant')

    return density


################################################################################
# rect the dot map
################################################################################
def rect_dot_map(dot_map):
    _, _, height, width = dot_map.shape
    peaks = torch.zeros_like(dot_map, requires_grad=True)

    coordinates = torch.where(dot_map[0][0] > 0)
    octopath = [[-1, -1], [-1, 0], [-1, 1], [0, -1],
                [0, 1], [1, -1], [1, 0], [1, 1]]

    for i in range(len(coordinates[0])):
        # [x0, y0] 为 warped_dot_map 上值不为 0 的点
        x0, y0 = coordinates[0][i], coordinates[1][i]
        # [x0, y0] 点的像素值
        val_0 = dot_map[0, 0, x0, y0]

        count = 0

        for offset in octopath:
            # [x, y] 为 warped_dot_map 上 [x0, y0] 的邻近点
            x, y = x0 + offset[0], y0 + offset[1]

            # 确保 [x, y] 在坐标范围内
            if 0 <= x < height and 0 <= y < width:
                # [x, y] 点的像素值
                val = dot_map[0, 0, x, y]
                if val_0 > val:
                    count += 1

        if count == 8:
            peaks[0, 0, x0, y0] = 1

    return peaks


def is_close(point_1, point_2):
    u_1, v_1 = torch.floor(point_1[0]), torch.floor(point_1[1])
    u_2, v_2 = torch.floor(point_2[0]), torch.floor(point_2[1])

    if torch.abs(u_1 - u_2) > 1 or torch.abs(v_1 - v_2) > 1:
        return False
    else:
        return True


def resize_grid(grid, shape):
    # type: (torch.Tensor, tuple) -> torch.Tensor
    grid = grid.permute(0, 3, 1, 2).contiguous()
    grid = F.interpolate(grid, size=shape,
                         mode='bilinear', align_corners=True)

    grid = grid.permute(0, 2, 3, 1).contiguous()

    return grid


################################################################################
# Resize the dot map with the given scale
################################################################################
def resize_dot_map(dot_map, scale_x, scale_y):
    h, w = dot_map.shape
    h, w = round(h * scale_x), round(w * scale_y)

    points_old = np.array(np.where(dot_map > 0), dtype=np.float64)
    points_new = np.zeros_like(points_old)
    points_new[0] = points_old[0] * scale_x
    points_new[1] = points_old[1] * scale_y
    points_new, points_old = points_new.transpose(), points_old.transpose()

    res = np.zeros([h, w], dtype=np.float64)

    for k in range(len(points_new)):
        i, j = np.floor(points_old[k] + 0.5)
        x, y = np.floor(points_new[k] + 0.5)

        x, y = min(x, h - 1), min(y, w - 1)

        res[int(x), int(y)] += dot_map[int(i), int(j)]

    return res


################################################################################
# average meter
################################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.cur_val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = float(self.sum / self.count)


class MultiAverageMeters(list):
    def __init__(self, num):
        super(MultiAverageMeters, self).__init__()
        self.num = num

        for i in range(num):
            self.append(AverageMeter())

    def reset(self):
        for i in range(self.num):
            self.__getitem__(i).reset()

    def update(self, values):
        for i in range(self.num):
            self.__getitem__(i).update(values[i])


################################################################################
# Timer
################################################################################
class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls


################################################################################
# Reverse image transformation
################################################################################
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


################################################################################
# Visualize density map and raw image simultaneously
################################################################################
def vis_den(im, density, save_path):
    dm_frame = plt.gca()
    plt.imshow(im)
    plt.imshow(density, 'jet', alpha=0.4)

    dm_frame.axes.get_yaxis().set_visible(False)
    dm_frame.axes.get_xaxis().set_visible(False)
    dm_frame.spines['top'].set_visible(False)
    dm_frame.spines['bottom'].set_visible(False)
    dm_frame.spines['left'].set_visible(False)
    dm_frame.spines['right'].set_visible(False)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)

    plt.close()


################################################################################
# Visualize dot map and raw image simultaneously
################################################################################
def vis_dot(im, dot_map, save_path):
    pts = np.asarray(np.where(dot_map > 0)).transpose()
    permutation = [1, 0]
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    pts[:] = pts[:, idx]

    dm_frame = plt.gca()
    plt.imshow(im)

    dm_frame.axes.get_yaxis().set_visible(False)
    dm_frame.axes.get_xaxis().set_visible(False)
    dm_frame.spines['top'].set_visible(False)
    dm_frame.spines['bottom'].set_visible(False)
    dm_frame.spines['left'].set_visible(False)
    dm_frame.spines['right'].set_visible(False)

    plt.scatter(pts[:, 0], pts[:, 1], s=5, c='r')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)

    plt.close()
