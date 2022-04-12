import torch.optim as optim

from misc.utilities import *
from models import BaseTrainer
from models.PSNet import PSNet
from misc.losses.attention_loss import computeAttentionLoss

################################################################################
# train model to generate density map
################################################################################
class Trainer(BaseTrainer):
    def __init__(self, cfg, data_loaders):
        super(Trainer, self).__init__(cfg, data_loaders)

        self.model = PSNet().float().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.optimizer.lr,
                                    weight_decay=cfg.optimizer.weight_decay)

        # loss function
        self.mseLoss = nn.MSELoss(reduction='sum').to(self.device)
        self.cosLoss = nn.CosineSimilarity(dim=1, eps=1e-8).to(self.device)

        self.meter = MultiAverageMeters(num=3)

    def epoch_train_step(self, i, data):
        img, _, gt_density, _ = data
        img = img.float().to(self.device)
        gt_density = gt_density.float().to(self.device)

        batch_size = img.size(0)
        self.optimizer.zero_grad()

        pr_density, attention_maps = self.model(img)

        # ----- MSE Loss -----
        density_loss = self.mseLoss(pr_density, gt_density) / batch_size

        # ----- Attention Loss -----
        weight_att = self.cfg.model.loss.lambda_1
        att_loss = computeAttentionLoss(attention_maps, self.cosLoss, batch_size)
        att_loss = att_loss * weight_att

        loss = density_loss + att_loss

        loss.backward()
        self.optimizer.step()

        self.meter.update([loss.item(), density_loss.item(), att_loss.item()])

        if (i + 1) % self.cfg.runner.print_freq == 0:
            logging.info('\tit: {:3d} | sum_loss: {:9.4f} | den_loss: {:9.4f}'
                         ' | att_loss: {:9.4f}'.
                         format(i + 1, self.meter[0].avg, self.meter[1].avg,
                                self.meter[2].avg))

        self.meter.reset()

    def epoch_val_step(self, data):
        img, _, gt_density = data
        img = img.float().to(self.device)

        pr_density, _ = self.model(img)

        # record real results and predicted results
        gt = torch.sum(gt_density).cpu().detach().numpy()
        pr = torch.sum(pr_density).cpu().detach().numpy()

        return gt, pr
