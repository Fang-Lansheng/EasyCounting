import torch
from torch.nn import Module
from .bregman_pytorch import sinkhorn


class OT_Loss(Module):
    def __init__(self, c_size, stride, norm_coord, device,
                 iter_in_ot=100, reg=10.0):
        super(OT_Loss, self).__init__()
        assert c_size % stride == 0

        self.c_size = c_size  # default: 256
        self.device = device  # default: cuda
        self.norm_coord = norm_coord  # default: 0
        self.iter_in_ot = iter_in_ot  # default: 100
        self.reg = reg  # default: 10.0
        # self.stride = stride  # default: 8

        # coordinate is same to image space,
        # set to constant since crop size is same
        self.coord = torch.arange(0, c_size, step=stride,
                                  dtype=torch.float32, device=device) \
                     + stride / 2
        self.density_size = self.coord.size(0)
        self.coord.unsqueeze_(0)  # [1, #cood]
        if self.norm_coord:  # default: False
            self.coord = self.coord / c_size * 2 - 1  # map to [-1, 1]
        self.output_size = self.coord.size(1)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, normed_density, unnormed_density, points):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        assert self.output_size == normed_density.size(2)
        loss = torch.zeros([1]).to(self.device)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0  # wasserstain distance
        for idx, im_points in enumerate(points):
            if len(im_points) > 0:
                # compute l2 square distance, it should be source target distance.
                # [#gt, #cood * #cood]
                if self.norm_coord:
                    # map to [-1, 1]
                    im_points = im_points / self.c_size * 2 - 1

                # [N, 1]
                x = im_points[:, 0].unsqueeze_(1)
                y = im_points[:, 1].unsqueeze_(1)

                # [#gt, #cood]
                x_dis = -2 * torch.matmul(x, self.coord) + \
                        x * x + self.coord * self.coord
                y_dis = -2 * torch.matmul(y, self.coord) + \
                        y * y + self.coord * self.coord

                y_dis.unsqueeze_(2)
                x_dis.unsqueeze_(1)

                dis = y_dis + x_dis
                # size of [#gt, #cood * #cood]
                dis = dis.view((dis.size(0), -1))

                source_prob = normed_density[idx][0].view([-1]).detach()
                target_prob = (torch.ones([len(im_points)]) / len(im_points))
                target_prob = target_prob.to(self.device)

                # use sinkhorn to solve OT, compute optimal beta.
                P, log = sinkhorn(target_prob, source_prob, dis, self.reg,
                                  maxIter=self.iter_in_ot, log=True)

                # size is the same as source_prob: [#cood * #cood]
                beta = log['beta']
                ot_obj_values += torch.sum(normed_density[idx] * beta.view(
                    [1, self.output_size, self.output_size]))

                # compute the gradient of OT loss to predicted density
                # (un-normed density).
                # im_grad = beta / source_count -
                #           < beta, source_density> / (source_count)^2
                source_density = unnormed_density[idx][0].view([-1]).detach()
                source_count = source_density.sum()

                # size of [#cood * #cood]
                im_grad_1 = source_count / (
                        source_count * source_count + 1e-8) * beta
                # size of 1
                im_grad_2 = (source_density * beta).sum() / (
                        source_count * source_count + 1e-8)
                im_grad = im_grad_1 - im_grad_2
                im_grad = im_grad.detach().view([
                    1, self.output_size, self.output_size])

                # Define loss = <im_grad, predicted density>.
                # The gradient of loss w.r.t predicted density is im_grad.
                loss += torch.sum(unnormed_density[idx] * im_grad)
                wd += torch.sum(dis * P).item()

        return loss, wd, ot_obj_values
