import torch.optim as optim

from models import BaseTrainer
from models.DM_Count import DM_Count
from misc.utilities import *
from misc.losses.ot_loss import OT_Loss


class Trainer(BaseTrainer):
    def __init__(self, cfg, data_loaders):
        super(Trainer, self).__init__(cfg, data_loaders)

        model = DM_Count().float()
        optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr,
                               weight_decay=cfg.optimizer.weight_decay)

        self.model = model.to(self.device)
        self.optimizer = optimizer

        if os.path.isfile(cfg.runner.resume):
            self.load_from_resume()

        # loss function
        self.maeLoss = nn.L1Loss().to(self.device)
        self.otLoss = OT_Loss(c_size=400, stride=8, norm_coord=0,
                              device=self.device, iter_in_ot=100, reg=10.0)
        self.tvLoss = nn.L1Loss(reduction='none').to(self.device)

        self.meter = MultiAverageMeters(num=4)

    def epoch_train_step(self, i, data):
        img, gt_dot_map, gt_density, gt_points = data
        img = img.float().to(self.device)
        gt_dot_map = gt_dot_map.float().to(self.device)

        x, x_normed = self.model(img)

        gt_points = [p.to(self.device) for p in gt_points]
        gt_counts = np.array([len(p) for p in gt_points], dtype=np.float32)
        gt_counts = torch.from_numpy(gt_counts).float().to(self.device)

        # ----- Count Loss -----
        count_loss = self.maeLoss(x.sum(1).sum(1).sum(1), gt_counts)

        # ----- OT Loss -----
        weight_ot = self.cfg.model.loss.lambda_1
        ot_loss, wd, ot_obj_value = self.otLoss(x_normed, x, gt_points)
        ot_loss, ot_obj_value = ot_loss * weight_ot, ot_obj_value * weight_ot

        # ----- TV Loss -----
        weight_tv = self.cfg.model.loss.lambda_2
        gt_dot_map_normed = gt_dot_map / (
                gt_counts.unsqueeze(1).unsqueeze(2).unsqueeze(3) + 1e-8)
        tv_loss = self.tvLoss(x_normed, gt_dot_map_normed).sum(1).sum(1).sum(1)
        tv_loss = (tv_loss * gt_counts).mean(0) * weight_tv

        loss = count_loss + ot_loss + tv_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.meter.update([loss.item(), count_loss.item(),
                           ot_loss.item(), tv_loss.item()])

        if (i + 1) % self.cfg.runner.print_freq == 0:
            logging.info('\tit: {:3d} | sum_loss: {:9.4f} | count_loss: {:9.4f} | '
                         'ot_loss: {:9.2e} | tv_loss: {:9.4f}'.
                         format(i + 1, self.meter[0].avg, self.meter[1].avg,
                                self.meter[2].avg, self.meter[3].avg))

        self.meter.reset()
