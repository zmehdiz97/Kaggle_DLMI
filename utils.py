import random
import numpy as np
import torch
import os
from torch import nn
import os
import datetime
from torch.nn import functional as F
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)
        
    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)


# Log
class Logger:
    def __init__(self, experiment_time, filename='log'):
        if not os.path.exists('./results'):
            os.mkdir('./results')
        self.logdir = os.path.join('./results', experiment_time)
        os.makedirs(self.logdir)
        self.logfile = os.path.join(self.logdir, filename)
        self.print_and_write("Experiment started at " + experiment_time)

    def print_and_write(self, log):
        print(log)
        with open(self.logfile, 'a') as f:
            f.write(log + '\n')

    def write(self, log):
        with open(self.logfile, 'a') as f:
            f.write(log + '\n')

Ng=6
y_shift = 2.035
def Kloss(x, target):
    x = Ng*torch.sigmoid(x.float()).view(-1) - 0.5
    target = target.float()
    return 1.0 - (2.0*((x-y_shift)*(target-y_shift)).sum() - 1e-3)/\
        (((x-y_shift)**2).sum() + ((target-y_shift)**2).sum() + 1e-3)

def hybrid_loss(x, target):
    crit1, crit2 = Kloss, nn.CrossEntropyLoss()
    loss_c = crit1(x[0].float(),target[:,:1].float())
    loss_caux = crit2(x[1].float(),target[:,1])
    return loss_c + 0.1*loss_caux

def both_loss(x, target):
    reg_pred, cls_pred, _ = x
    crit1, crit2 = nn.MSELoss(), nn.CrossEntropyLoss()
    loss_reg = crit1(reg_pred.float(),target.float())
    #loss_cls = crit2(cls_pred.float(),target)
    return loss_reg #+ loss_cls
    
def mse_loss(x, target):
    crit = nn.MSELoss()
    return crit(x, target[:,None].float())