import torch.optim as optim

from misc.utilities import *
from models import BaseTrainer
from models.STDNet import STDNet


class Trainer(BaseTrainer):
    def __init__(self, cfg, data_loaders):
        super(Trainer, self).__init__(cfg, data_loaders)

        model = STDNet(num_blocks=cfg.model.num_blocks,
                       use_bn=cfg.model.use_bn).float()

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

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        logging.info('\tTrainable Parameters: %.3fM' % parameters)

    def epoch_train_step(self, i, data):
        imgs, dens = data
        imgs = imgs.float().to(self.device)     # [B, T, _, H, W]
        dens = dens.float().to(self.device)     # [B, T, 1, H, W]

        outs = self.model(imgs)                 # [B, T, 1, H, W]

        ''' MSE Loss '''
        mse_loss = self.mseLoss(outs, dens)

        loss = mse_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.meter.update([loss.item(), mse_loss.item()])

        if (i + 1) % self.cfg.runner.print_freq == 0:
            logging.info('\tit: {:4d} | sum_loss: {:10.4f} | mse_loss: {:10.4f}'.
                         format(i + 1, self.meter[0].avg, self.meter[1].avg))

        self.meter.reset()

    def epoch_val_step(self, data):
        imgs, dens = data
        imgs = imgs.float().to(self.device)

        outs = self.model(imgs)     # [1, time_step, 1, H, W]

        # record real results and predicted results
        gt = dens[0].sum(-1).sum(-1).sum(-1)        # [time_step]
        pr = outs[0].sum(-1).sum(-1).sum(-1)        # [time_step]
        gt = gt.cpu().detach().numpy()
        pr = pr.cpu().detach().numpy()

        return gt, pr

    def epoch_val(self):
        self.model.eval()

        mae_sum, mse_sum = 0.0, 0.0
        N = 0

        with torch.no_grad():
            for data in self.val_loader:
                gt, pr = self.epoch_val_step(data)  # [time_step, 1]

                time_step = gt.shape[0]
                for i in range(time_step):
                    N += 1
                    mae_sum += np.abs(gt[i] - pr[i])
                    mse_sum += (gt[i] - pr[i]) ** 2

        self.CURR_MAE = mae_sum / N
        self.CURR_MSE = np.sqrt(mse_sum / N)


