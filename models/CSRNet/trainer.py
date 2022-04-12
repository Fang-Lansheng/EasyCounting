import torch.optim as optim

from models import BaseTrainer
from models.CSRNet import CSRNet
from misc.utilities import *
from misc.losses.uniform_loss import *


class Trainer(BaseTrainer):
    def __init__(self, cfg, data_loaders):
        super(Trainer, self).__init__(cfg, data_loaders)

        model = CSRNet().float()

        if cfg.optimizer.type == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.lr,
                                  momentum=cfg.optimizer.momentum,
                                  weight_decay=cfg.optimizer.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr,
                                   weight_decay=cfg.optimizer.weight_decay)

        self.model = model.to(self.device)
        self.optimizer = optimizer

        if os.path.isfile(cfg.runner.resume):
            self.load_from_resume()

        # loss function
        self.maeLoss = nn.L1Loss().to(self.device)
        self.mseLoss = nn.MSELoss(reduction='sum').to(self.device)

        self.meter = MultiAverageMeters(num=2)

    def epoch_train_step(self, i, data):
        img, _, gt_density, _ = data
        img = img.float().to(self.device)
        gt_density = gt_density.float().to(self.device)

        x = self.model(img)

        ''' MSE Loss '''
        mse_loss = self.mseLoss(x, gt_density) / img.shape[0]

        loss = mse_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.meter.update([loss.item(), mse_loss.item()])

        if (i + 1) % self.cfg.runner.print_freq == 0:
            logging.info('\tit: {:3d} | sum_loss: {:9.4f} | mse_loss: {:9.4f}'.
                         format(i + 1, self.meter[0].avg, self.meter[1].avg))

        self.meter.reset()

    def epoch_val_step(self, data):
        img, _, gt_density = data
        img = img.float().to(self.device)

        pr_density = self.model(img)

        # record real results and predicted results
        pr = torch.sum(pr_density).cpu().detach().numpy()
        gt = torch.sum(gt_density).cpu().detach().numpy()

        return gt, pr
