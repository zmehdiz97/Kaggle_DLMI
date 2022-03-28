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
from utils import seed_everything

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def get_aug(p=0.2, train=True):
    trans = [
        transforms.ToTensor(),
        transforms.Normalize((0.63, 0.41, 0.59), (0.48, 0.46, 0.43))
    ]
    aug = [
        transforms.RandomHorizontalFlip(p),
        transforms.RandomVerticalFlip(p),
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.RandomAffine(degrees=(-15, 15), 
                                    translate=(0, 0.05), 
                                    scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1))
        ]), p=p)
    ]
    if train:
        return transforms.Compose(trans+aug)
    else:
        return transforms.Compose(trans)

class CustomDataset(Dataset):
    def __init__(self, df, N, path, train=True, transforms=None):
        self.df = df.loc[df.train == 1].copy() if train else df.loc[df.train == 0].copy()
        self.df = self.df.reset_index(drop=True)
        self.path = path
        self.transforms = transforms
        self.N=N
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        labels = self.df.iloc[idx][['isup_grade']].astype(np.int).values
        #provider = self.df.iloc[idx].data_provider
        
        image_id = self.df.iloc[idx].image_id
        imgs = []
        for i in range(self.N):
            img = Image.open(os.path.join(self.path,image_id+'_'+str(i)+'.png'))
            #img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN,image_id+'_'+str(i)+'.png')), cv2.COLOR_BGR2RGB)
            #img = 255 - img
            if self.transforms is not None:
                img = self.transforms(img)
            imgs.append(img)
        #imgs = [img2tensor((img/255.0 - mean)/std,np.float32) for img in imgs]

        return torch.stack(imgs,0), labels

if __name__ == "__main__":
    print(torch.cuda.is_available())
    SEED = 43
    TRAIN = 'data/train_128x128'
    LABELS = 'data/train.csv'
    N=128
    seed_everything(SEED)

    df = pd.read_csv(LABELS).set_index('image_id')
    files = sorted(set([p[:32] for p in os.listdir(TRAIN)]))
    df.gleason_score = df.gleason_score.replace('negative','0+0')
    df = df.loc[files]
    df = df.reset_index()
    train_test = np.random.choice([0, 1], size=len(df), p=[.2, .8])
    df['train'] = train_test
    df.to_csv(LABELS.replace('train', 'trainv2'), index=False)


    ds = CustomDataset(df,N, TRAIN, train=True, transforms=get_aug(.5))
    x,y = ds[5]
    print(x.shape)
    x = x.reshape(N, 128,128,3).numpy()
    #t = 255 - ((x.permute(0,2,3,1)*std + mean)*255.0).byte()
    plt.figure(figsize=(16,32))
    for i in range(len(x)):
        plt.subplot(16,8,i+1)
        plt.imshow(x[i])
        plt.axis('off')
        plt.subplots_adjust(wspace=None, hspace=None)
