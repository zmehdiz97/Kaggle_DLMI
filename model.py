import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from efficientnet_pytorch import EfficientNet
from utils import AdaptiveConcatPool2d
from sklearn.metrics import cohen_kappa_score
import numpy as np
from functools import partial
import scipy as sp
from torchvision import models 
from pooling import AdaptiveConcatPool2d_Attention



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
    def __init__(self, backbone='efficientnet-b0', p=0.5, mode='classification'):
        super(effnet, self).__init__()
        assert mode in ['classification', 'regression', 'hybrid']
        self.mode = mode
        if self.mode == 'classification': n = 6
        elif self.mode == 'regression': n = 1
        else: n = 11
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
        if self.mode == 'hybrid':
          return x[:,:1],x[:,1:]
        else: 
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
    

class MyResNet34(nn.Module):
    def __init__(self, pretrained=True, dropout=0.4, mode = 'classification'):
        super(MyResNet34, self).__init__()
        assert mode in ['classification', 'regression', 'hybrid']
        self.mode = mode
        if self.mode == 'classification': n = 6
        elif self.mode == 'regression': n = 1
        else: n = 11
        model = models.resnet34(pretrained)
        num_ftrs = model.fc.in_features
        self.enc = nn.Sequential(*list(model.children())[:-2])
        
        self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),
                            nn.Linear(2*num_ftrs,512),nn.Mish(),nn.GroupNorm(32,512),
                            nn.Dropout(dropout),nn.Linear(512,n))
        

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
      
class Attention_Model(nn.Module):
    '''
        Pytorch Model used in PANDA Challenge
    '''
    def __init__(self,dropout=0.4,scale_op=True,gated=False):
        super().__init__()
        self.scale_op=scale_op
        enet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                              'nvidia_efficientnet_b0', pretrained=True)
        back_feature = enet.classifier.fc.in_features
        self.base_model = nn.Sequential(*list(enet.children())[:-1])
        
        self.avg_pool=nn.AdaptiveAvgPool2d(1)

        self.attention=AdaptiveConcatPool2d_Attention(in_ch=back_feature,hidden=512,dropout=dropout,gated=gated)

        self.reg_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2*back_feature,1,bias=True),
        )
        self.cls_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2*back_feature,6,bias=True),
        )


    def forward(self,x):
        # x [bs,n,3,h,w]
        B,N,C,H,W=x.shape
        x=x.view(-1,C,H,W)
        x=self.base_model(x)
 
        x=self.avg_pool(x).view(x.size(0),-1)

        x=x.view(B,N,-1)
        x,A=self.attention(x)

        reg_pred=self.reg_head(x).view(-1)
        if self.scale_op:
            reg_pred=7.*torch.sigmoid(reg_pred)-1.
        cls_pred=self.cls_head(x)
        return reg_pred,cls_pred,A
    
