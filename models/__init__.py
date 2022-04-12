from misc.utilities import *


class BaseTrainer(nn.Module):
    def __init__(self, cfg, data_loaders):
        super(BaseTrainer, self).__init__()

        self.cfg = cfg
        self.train_loader, self.val_loader = data_loaders

        if torch.cuda.is_available():
            # gpus = str(cfg.runner.device)
            # os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.epoch = 0
        self.start_epoch = 0
        self.BEST_MAE = 300
        self.BEST_MSE = 300

        self.CURR_MAE = self.BEST_MAE
        self.CURR_MSE = self.BEST_MSE

        self.timer = {'train': Timer(), 'val': Timer()}

    def run(self):
        for epoch in range(self.start_epoch, self.cfg.runner.max_epochs):
            # ----- training -----
            self.timer['train'].tic()
            self.epoch = epoch

            adjust_learning_rate(self.optimizer, epoch)

            logging.info('> Epoch: {:4d}/{:4d} {:s}'.format(
                self.epoch + 1, self.cfg.runner.max_epochs, '-' * 80))

            self.epoch_train()
            self.save_model(self.epoch, 'ckpt.pth')

            self.timer['train'].toc()
            logging.info('  - [Train] cost time: {:5.1f}s | lr: {:.4e}'.format(
                self.timer['train'].diff, self.optimizer.param_groups[0]['lr']))

            # ----- validation -----
            if (epoch + 1) % self.cfg.runner.val_freq == 0:
                self.timer['val'].tic()

                self.epoch_val()
                if self.CURR_MAE < self.BEST_MAE:
                    self.BEST_MAE = self.CURR_MAE
                    self.BEST_MSE = self.CURR_MSE
                    self.save_model(epoch, 'ckpt_best.pth')

                self.timer['val'].toc()
                logging.info('  - [Val]   cost time: {:5.1f}s | MAE: {:6.2f}, MSE: {:6.2f} '
                             '(BEST: {:6.2f}/{:6.2f})'.
                             format(self.timer['val'].diff, self.CURR_MAE, self.CURR_MSE,
                                    self.BEST_MAE, self.BEST_MSE))

    def epoch_train(self):
        self.model.train()

        for (i, data) in enumerate(self.train_loader):
            self.epoch_train_step(i, data)

    def epoch_train_step(self, i, data):
        pass

    def epoch_val(self):
        self.model.eval()

        mae_sum, mse_sum = 0.0, 0.0
        N = self.val_loader.__len__()

        with torch.no_grad():
            for data in self.val_loader:
                gt, pr = self.epoch_val_step(data)

                mae_sum += np.abs(gt - pr)
                mse_sum += np.abs(gt - pr) ** 2

        self.CURR_MAE = mae_sum / N
        self.CURR_MSE = np.sqrt(mse_sum / N)

    def epoch_val_step(self, data):
        img, gt_dot_map, gt_density = data
        img = img.float().to(self.device)
        # gt_dot_map = gt_dot_map.float().to(self.device)
        # gt_density = gt_density.float().to(self.device)

        pr_density, _ = self.model(img)

        # record real results and predicted results
        gt = torch.sum(gt_dot_map).cpu().detach().numpy()
        pr = torch.sum(pr_density).cpu().detach().numpy()

        return gt, pr

    def load_from_resume(self):
        logging.info('\t==> Resuming from checkpoint: {:s}'.format(
            self.cfg.runner.resume))

        state = torch.load(self.cfg.runner.resume)
        self.model.load_state_dict(state['net'])
        self.optimizer.load_state_dict(state['optim'])

        self.start_epoch = state['epoch']
        self.BEST_MAE = state['mae']
        self.BEST_MSE = state['mse']

    def save_model(self, epoch, name):
        state = {'net': self.model.state_dict(),
                 'optim': self.optimizer.state_dict(),
                 'epoch': epoch,
                 'mae': self.BEST_MAE,
                 'mse': self.BEST_MSE,
                 'cfg': self.cfg}
        torch.save(state, os.path.join(self.cfg.runner.ckpt_dir, name))

    def img_transform(self, img):
        mean = self.cfg.dataset.img_norm_cfg.mean
        std = self.cfg.dataset.img_norm_cfg.std
        t = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

        return t(img)

    def img_restoration(self, img):
        mean = self.cfg.dataset.img_norm_cfg.mean
        std = self.cfg.dataset.img_norm_cfg.std
        t = transforms.Compose([de_normalize(mean, std),
                                transforms.ToPILImage()])

        return t(img)


class BaseTester(nn.Module):
    def __init__(self, ckpt, cfg, val_loader, vis_options):
        super(BaseTester, self).__init__()

        self.ckpt = ckpt
        self.cfg = cfg
        self.val_loader = val_loader
        self.vis_options = vis_options

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.MAE = 0.
        self.MSE = 0.
        self.timer = Timer
        self.restore_transform = transforms.Compose([
            de_normalize(mean=self.cfg.dataset.img_norm_cfg.mean,
                         std=self.cfg.dataset.img_norm_cfg.std),
            transforms.ToPILImage()])

        self.vis_dirs = self.create_dir()

    def run(self):
        pass

    def create_dir(self):
        ckpt_dir = self.cfg.runner.ckpt_dir
        infer_dir = osp.join(ckpt_dir, 'inference')
        os.makedirs(infer_dir, exist_ok=True)

        vis_dirs = {}

        for k, v in self.vis_options.items():
            if v:
                dirname = osp.join(infer_dir, k)
                os.makedirs(dirname, exist_ok=True)
                vis_dirs[k] = dirname

        return vis_dirs

    def visualization(self, img_name, img, gt, pr_density, by_product=None):
        gt_dot_map, gt_density = gt

        # save img
        if self.vis_options['img_raw']:
            img_raw = self.restore_transform(img.cpu().squeeze())
            img_raw.save(osp.join(self.vis_dirs['img_raw'], img_name))

        # save dot map
        if self.vis_options['gt_dot_map']:
            gt_dot_map = gt_dot_map.cpu().detach().numpy()
            plot_density(gt_dot_map, self.vis_dirs['gt_dot_map'], img_name)

        # save density map
        if self.vis_options['gt_density']:
            gt_density = gt_density.cpu().detach().numpy()
            plot_density(gt_density, self.vis_dirs['gt_density'], img_name)

        # save prediction
        if self.vis_options['pr_density']:
            pr_density = pr_density.cpu().detach().numpy()
            plot_density(pr_density, self.vis_dirs['pr_density'], img_name)
