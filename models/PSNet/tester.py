from misc.utilities import *
from models import BaseTester
from models.PSNet import PSNet


class Tester(BaseTester):
    def __init__(self, ckpt, cfg, val_loader, vis_options):
        super(Tester, self).__init__(ckpt, cfg, val_loader, vis_options)

        model = PSNet().float()
        self.model = model.to(self.device)
        self.model.load_state_dict(self.ckpt['net'])

    def run(self):
        self.model.eval()

        count = 0
        mae_, mse_ = 0., 0.

        logging.info('\t{:>4s} | {:<10s} | {:>4s} | {:>12s} {:>8s}'.format(
            'no.', 'name', 'gt', 'prediction', 'diff'))

        with torch.no_grad():
            for data in self.val_loader:
                img_name = osp.basename(self.val_loader.dataset.img_path_list[count])
                count += 1

                img, gt_dot_map, gt_density = data
                img = img.float().to(self.device)

                pr_density = self.model(img)

                # record real results and predicted results
                gt = torch.sum(gt_dot_map).cpu().detach().numpy()
                pr = torch.sum(pr_density).cpu().detach().numpy()

                self.MAE += np.abs(gt - pr)
                self.MSE += np.abs(gt - pr) ** 2

                self.visualization(img_name, img,
                                   [gt_dot_map, gt_density], pr_density)

                logging.info('\t{:4d} | {:<10s} | {:>4d} | {:>12.2f} ({:>8.2f}) | '.
                             format(count, img_name.split('.')[0], int(gt), pr, pr - gt))

        self.MAE = self.MAE / count
        self.MSE = np.sqrt(self.MSE / count)

        mae_ = mae_ / count
        mse_ = np.sqrt(mse_ / count)

        logging.info('{} {}'.format('> END TESTING     ', '-' * 80))
        logging.info('\t[w/o warping] MAE: {:6.2f}\tMSE: {:6.2f}'.format(mae_, mse_))
        logging.info('\t[w/  warping] MAE: {:6.2f}\tMSE: {:6.2f}'.format(self.MAE, self.MSE))
