from misc.utilities import *
from models import BaseTester
from models.STDNet import STDNet


class Tester(BaseTester):
    def __init__(self, ckpt, cfg, val_loader, vis_options):
        super(Tester, self).__init__(ckpt, cfg, val_loader, vis_options)

        model = STDNet(num_blocks=cfg.model.num_blocks,
                       use_bn=cfg.model.use_bn).float()

        self.model = model.to(self.device)
        self.model.load_state_dict(self.ckpt['net'])

    def run(self):
        self.model.eval()

        count_0 = 0
        count = 0
        logging.info('\t{:>4s} | {:<10s} | {:>4s} | {:>12s} {:>8s}'.format(
            'no.', 'name', 'gt', 'prediction', 'diff'))

        with torch.no_grad():
            for data in self.val_loader:
                img_path_list = self.val_loader.dataset.img_path_list[count_0]
                count_0 += 1
                imgs, dens = data
                T = len(img_path_list)

                imgs = imgs.float().to(self.device)  # [B, T, _, H, W]
                dens = dens.float().to(self.device)  # [B, T, 1, H, W]

                outs = self.model(imgs)

                for i in range(T):
                    img_name = osp.basename(img_path_list[i])
                    count += 1

                    img = imgs[0][i].unsqueeze(0)
                    den = dens[0][i].unsqueeze(0)
                    out = outs[0][i].unsqueeze(0)

                    # record real results and predicted results
                    gt = torch.sum(den).cpu().detach().numpy()
                    pr = torch.sum(out).cpu().detach().numpy()

                    self.MAE += np.abs(gt - pr)
                    self.MSE += np.abs(gt - pr) ** 2

                    self.visualization(img_name, img, [den, den], out)

                    logging.info('\t{:4d} | {:<10s} | {:>4d} | {:>12.2f} ({:>8.2f}) | '.
                                 format(count, img_name.split('.')[0], int(gt + 0.5), pr, pr - gt))

        self.MAE = self.MAE / count
        self.MSE = np.sqrt(self.MSE / count)

        logging.info('{} {}'.format('> END TESTING     ', '-' * 80))
        logging.info('\t[Test Result] MAE: {:6.2f}\tMSE: {:6.2f}'.format(self.MAE, self.MSE))

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
            img_raw = self.restore_transform(img.cpu().squeeze())
            pr_density = pr_density.cpu().detach().numpy()
            # plot_density(pr_density, self.vis_dirs['pr_density'], img_name)
            plot_pred(pr_density, img_raw, self.vis_dirs['pr_density'], img_name)


def plot_pred(pr_density, img, dm_dir, img_name, alpha=0.5):
    assert osp.isdir(dm_dir)

    dm = pr_density[0, 0, :, :]
    dm = dm / np.max(dm + 1e-20)

    dm_frame = plt.gca()
    plt.imshow(img, alpha=1-alpha)
    plt.imshow(dm, 'jet', alpha=alpha)

    dm_frame.axes.get_yaxis().set_visible(False)
    dm_frame.axes.get_xaxis().set_visible(False)
    dm_frame.spines['top'].set_visible(False)
    dm_frame.spines['bottom'].set_visible(False)
    dm_frame.spines['left'].set_visible(False)
    dm_frame.spines['right'].set_visible(False)

    dm_file_name = img_name.split('.')[0]
    plt.savefig(osp.join(dm_dir, dm_file_name), bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
