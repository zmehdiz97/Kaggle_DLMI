import torch
from torch import nn
from utils import AdaptiveConcatPool2d
class Model(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl',n=6,ps=0.2):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        nc = list(m.children())[-1].in_features
        self.enc = nn.Sequential(*list(m.children())[:-2])
        self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),
                                  nn.Linear(2*nc,512),nn.Mish(),nn.GroupNorm(32,512),
                                  nn.Dropout(ps),nn.Linear(512,n))
        
    def forward(self, x):
        shape = x.shape
        n = shape[1]
        x = x.view(-1,shape[2],shape[3],shape[4])
        x = self.enc(x)
        
        shape = x.shape
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
          .view(-1,shape[1],shape[2]*n,shape[3])
        x = self.head(x)
        return x