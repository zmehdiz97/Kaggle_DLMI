import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from utils import AdaptiveConcatPool2d
from sklearn.metrics import cohen_kappa_score
import numpy as np
from functools import partial
import scipy as sp

class Model(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl',ps=0.4, mode='classification'):
        super().__init__()
        assert mode in ['classification', 'regression', 'hybrid']
        self.mode = mode
        if self.mode == 'classification': n = 6
        elif self.mode == 'regression': n = 1
        else: n = 11
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
        if self.mode == 'hybrid':
          return x[:,:1],x[:,1:]
        else: 
          return x
      
class effnet(nn.Module):
    def __init__(self, backbone='efficientnet-b0', n=6, p=0.4, mode='classification'):
        super(effnet, self).__init__()
        #enet = EfficientNet.from_pretrained('efficientnet-b0',
        #                                          num_classes=n)
        enet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                              'nvidia_efficientnet_b0', pretrained=True)

        nc = enet.classifier.fc.in_features
        self.enc = nn.Sequential(*list(enet.children())[:-1])
        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                             nn.Flatten(),
                                             nn.Linear(2*nc,512),
                                             nn.Mish(),
                                             nn.GroupNorm(32,512),
                                             nn.Dropout(p),
                                             nn.Linear(512,n))


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
      

def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')
  
class OptimizedRounder():
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5
        return X_p

    def coefficients(self):
        return self.coef_['x']
