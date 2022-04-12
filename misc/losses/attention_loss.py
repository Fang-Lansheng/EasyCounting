import torch


def computeAttentionLoss(attention_maps, cosLoss, batch_size):
    attention_loss = 0.
    for attention_map in attention_maps:
        attention_map_sum = attention_map[:, 0:1] + \
                            attention_map[:, 1:2] + \
                            attention_map[:, 2:3] + \
                            attention_map[:, 3:4]
        attention_loss_temp = 0.
        for i in range(4):
            item_1 = attention_map[:, i:(i + 1)].contiguous().view(batch_size, -1)
            item_2 = ((attention_map_sum - attention_map[:, i:(i + 1)]) / 3). \
                contiguous().view(batch_size, -1)
            attention_loss_temp += torch.sum(cosLoss(item_1, item_2)) / batch_size
        attention_loss += (attention_loss_temp / 4)
    attention_loss /= len(attention_maps)

    return attention_loss


