import random
import numpy as np
import torch
import os
from torch import nn
import os
import datetime

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
