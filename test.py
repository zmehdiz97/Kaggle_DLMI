import os
import cv2
import skimage.io
from tqdm import tqdm
import zipfile
import numpy as np

TRAIN = 'data/train_label_masks'
OUT_TRAIN = 'data/train_label_masks_128x128'
sz = 128 #tile size
N = 128  #number of tiles
L = 0    #[0-2] tiff layer
A = 0    #[0-3] if not equal to 0, adds extra sz//2 padding to generate new tiles

os.makedirs(OUT_TRAIN, exist_ok=True)

def tile(img):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    padval = 255
    print(shape, np.all(img == 0), np.sum(img > 0))

    if A == 1:
        img = np.pad(img,[[pad0//2 + sz//2,pad0-pad0//2 + sz//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=padval)
    if A == 2:
        img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2 + sz//2,pad1-pad1//2 + sz//2],[0,0]],
                constant_values=padval)
    if A == 3:
        img = np.pad(img,[[pad0//2 + sz//2,pad0-pad0//2 + sz//2],[pad1//2 + sz//2,pad1-pad1//2 + sz//2],[0,0]],
                constant_values=padval)
    else:
        img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=padval)

    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=padval)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    #for i in range(len(img)):
    #    result.append({'img':img[i], 'idx':i})
    return img


fnames = os.listdir(OUT_TRAIN)# + os.listdir(OUT_MASKS)
rdy_dict = {}
for name in fnames:
    n = name.split('_')[0]
    if n in rdy_dict:
        rdy_dict[n] += 1
    else: rdy_dict[n] = 1

names = [name[:-5] for name in os.listdir(TRAIN)]
def convert(name):
    if name in rdy_dict and rdy_dict[name] == N:#2*N:
        return [],[]
    x_tot,x2_tot = [],[]
    img = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[L]
    print(type(img))
    rgb_weights = [0.2989, 0.5870, 0.1140]
    img = np.dot(img[...,:3], rgb_weights)
    print(type(img))
    mask = None
    tiles = tile(img)
    for idx,img in enumerate(tiles):
        #img,idx = t['img'],t['idx']
        x_tot.append((img/255.0).reshape(-1,3).mean(0))
        x2_tot.append(((img/255.0)**2).reshape(-1,3).mean(0)) 
        #if read with PIL RGB turns into BGR
        cv2.imwrite(os.path.join(OUT_TRAIN,f'{name}_{idx}.png'),cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return x_tot,x2_tot

results = [convert(name) for name in tqdm(names)]

x_tot,x2_tot = [],[]
for r in results:
    x_tot += r[0]
    x2_tot += r[1]

#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', np.sqrt(img_std))