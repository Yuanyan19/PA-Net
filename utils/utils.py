import torch
import numpy as np
# `pip install thop`
from thop import profile
from thop import clever_format


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating mis-alignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):

    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('#'*20, '\n[Statistics Information]\nFLOPs: {}\nParams: {}\n'.format(flops, params), '#'*20)


#################################################################################    
    
def one_hot_segmentation(label, num_cls):
    batch_size = label.size(0)
    label = label.long()
    out_tensor = torch.zeros(batch_size, num_cls, *label.size()[2:]).to(label.device)
    out_tensor.scatter_(1, label, 1)

    return out_tensor




def dice_coef_2d(pred, target):
    pred = torch.argmax(pred, dim=1, keepdim=True).float()
    target = torch.gt(target, 0.5).float()
    n = target.size(0)
    smooth = 1e-4
    
    target = target.view(n, -1)
    pred = pred.view(n, -1)
    intersect = torch.sum(target * pred, dim=-1)
    dice = (2 * intersect + smooth) / (torch.sum(target, dim=-1) + torch.sum(pred, dim=-1) + smooth)
    dice = torch.mean(dice)

    return dice