import os
from radam import *
#from csvlogger import *
from mish import *
import cv2
#from albumentations import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import seed_everything

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))
#L1 128x128x128 mean: [0.79718421 0.58146681 0.72599565] , std: [0.3969224  0.48599503 0.3936849 ]
#STATS = ((0.79718421 0.58146681 0.72599565) , std: [0.3969224  0.48599503 0.3936849 ]
#STATS = ((1-0.87622766, 1-0.75070891, 1-0.83482135) ,(0.39165375, 0.51765024, 0.41787194)) #(0.63, 0.41, 0.59), (0.48, 0.46, 0.43)
STATS = ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

def get_aug(p=0.8, train=True):
    if not train: p=0
    aug = A.Compose([
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.RandomRotate90(p=p),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.3*p, 
                            border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([
                A.HueSaturationValue(10,15,10),
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(),            
            ], p=0.35*p),
            A.Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
    ])
    return aug
#def get_aug(p=0.2, train=True):
#    trans = [
#        transforms.ToTensor(),
#        transforms.Normalize(STATS[0], STATS[1])
#    ]
#    aug = [
#        transforms.RandomHorizontalFlip(p),
#        transforms.RandomVerticalFlip(p),
#        
#        #transforms.RandomApply(torch.nn.ModuleList([
#        #    transforms.RandomRotation(degrees=(90, 90)),
#        #    transforms.RandomAffine(degrees=(-15, 15), 
#        #                            translate=(0, 0.05), 
#        #                            scale=(0.8, 1.2)),
#        #    transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1))
#        #]), p=p)
#    ]
#    if train:
#        return transforms.Compose(trans+aug)
#    else:
#        return transforms.Compose(trans)

class CustomDataset(Dataset):
    def __init__(self, df, N, path, fold, train=True, transforms=None, mode='classification'):
        assert mode in ['classification', 'regression', 'hybrid', 'both', 'binning']
        self.mode = mode
        self.df = df.loc[df.split != fold].copy() if train else df.loc[df.split == fold].copy()
        self.df = self.df.reset_index(drop=True)
        self.path = path
        self.transforms = transforms
        self.N=N
        self.train = train
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mode == 'binning':
            labels = np.zeros(5).astype(np.float32)
            labels[:self.df.iloc[idx][['isup_grade']].astype(int).values[0]] = 1.
        elif self.mode=='hybrid':
            labels = self.df.iloc[idx][['isup_grade','score']].astype(int).values
        else:
            labels = self.df.iloc[idx][['isup_grade']].astype(int).values

        #provider = self.df.iloc[idx].data_provider
        
        image_id = self.df.iloc[idx].image_id
        imgs = []
        ntiles = 36
        n = self.N if self.train else 2*self.N
        if self.train:  ids = random.choices(range(ntiles),k=n)
        else: ids = range(min(n,ntiles))
        ids = random.choices(range(ntiles),k=n)
        for i in ids:
            #img = Image.open(os.path.join(self.path,image_id+'_'+str(i)+'.png'))
            img = cv2.cvtColor(cv2.imread(os.path.join(self.path,image_id+'_'+str(i)+'.png')), cv2.COLOR_BGR2RGB)
            img = 255 - img
            if self.transforms is not None:
                #img = self.transforms(img)
                img = self.transforms(image=img)['image']
            imgs.append(img/255.0)
        #imgs = [img2tensor((img/255.0 - mean)/std,np.float32) for img in imgs]

        return torch.stack(imgs,0), labels
    
class TestDataset(Dataset):
    def __init__(self, df, N, path):
        self.df = df
        self.df = self.df.reset_index(drop=True)
        self.path = path
        self.N=N
        self.transforms = A.Compose(A.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                    ToTensorV2())
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        #provider = self.df.iloc[idx].data_provider
        
        image_id = self.df.iloc[idx].image_id
        imgs = []
        
        for i in range(self.N):
            print(os.path.join(self.path,image_id+'_'+str(i)+'.png'))
            #img = Image.open(os.path.join(self.path,image_id+'_'+str(i)+'.png'))
            img = cv2.cvtColor(cv2.imread(os.path.join(self.path,image_id+'_'+str(i)+'.png')), cv2.COLOR_BGR2RGB)
            img = 255 - img
            img = self.transforms(image=img)['image']
            imgs.append(img)
        #imgs = [img2tensor((img/255.0 - mean)/std,np.float32) for img in imgs]

        return torch.stack(imgs,0), image_id

if __name__ == "__main__":
    print(torch.cuda.is_available())
    SEED = 43
    TRAIN = 'data/train_128x128'
    LABELS = 'data/train.csv'
    N=128
    nfolds = 4
    seed_everything(SEED)
    
    df = pd.read_csv(LABELS).set_index('image_id')
    files = sorted(set([p[:32] for p in os.listdir(TRAIN)]))
    df.gleason_score = df.gleason_score.replace('negative','0+0')
    df = df.loc[files]
    df = df.reset_index()
    splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
    splits = list(splits.split(df,df.isup_grade))
    folds_splits = np.zeros(len(df)).astype(np.int)
    for i in range(nfolds):
        folds_splits[splits[i][1]] = i
    df['split'] = folds_splits
    Ng, Ns = df.nunique()[2], df.nunique()[3]
    score_map = {s:i for i,s in enumerate(df.gleason_score.unique())}
    df['score'] = df.gleason_score.map(score_map)
    df.to_csv(LABELS.replace('train', 'trainv3'), index=False)
    print(df.isup_grade.mean(), df.nunique()[2], df.nunique()[3])


    '''
    df = pd.read_csv(LABELS).set_index('image_id')
    files = sorted(set([p[:32] for p in os.listdir(TRAIN)]))
    df.gleason_score = df.gleason_score.replace('negative','0+0')
    df = df.loc[files]
    df = df.reset_index()
    train_test = np.random.choice([0, 1], size=len(df), p=[.2, .8])
    df['train'] = train_test
    '''