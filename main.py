import os
from argparse import ArgumentParser
from tqdm import tqdm
import datetime

import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, dataset
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
import numpy as np

from utils import seed_everything, Logger, Kloss, hybrid_loss, both_loss, mse_loss
from dataset import get_aug, CustomDataset
from model import Model, effnet, OptimizedRounder, MyResNet34, Attention_Model, Binning_Attention_Model




###########################################################
########### DEFINE MAIN FUNCTION
###########################################################

def main():

    # Check device
    global use_gpu
    use_gpu = torch.cuda.is_available()

    # Get arguments from console
    parser = ArgumentParser()
    parser.add_argument("function", nargs="?", choices=["train", "test"])
    args, sub_args = parser.parse_known_args()
    print(args.function)
    # If user wants to train
    if args.function == "train":
        parser = ArgumentParser()
        parser.add_argument('--labels', type=str, default='data/trainv3.csv', help='Path to CSV with image ids and labels')
        parser.add_argument('--path', type=str, default='data/train_L1_256x256', help='Path to image tiles')
        parser.add_argument('--nepochs', type=int, default=100, help='Number of epochs')
        parser.add_argument('--bs', type=int, default=2, help='batch size')
        parser.add_argument('--fold', type=int, default=0, help='index fold (should be referred in csv)')

        parser.add_argument('--resume_from', default=None, help='Path to checkpoint')
        parser.add_argument('--mode', '-m', help='model mode: classification, regression, hybrid',
                            default='classification', type=str,
                            choices=['classification', 'regression', 'hybrid', 'both', 'binning'])
        args = parser.parse_args(sub_args)
        train(args.nepochs, args.bs, args.labels, args.path, args.resume_from, args.fold, args.mode)

    ## If user wants to test
    #elif args.function == "test":
    #    parser = ArgumentParser()
    #    parser.add_argument('PATH_TO_CHECKPOINT')
    #    parser.add_argument('PATH_TO_TEST_SET')
    #    args = parser.parse_args(sub_args)
    #    if not os.path.exists('./outputs'): os.mkdir('./outputs')
    #    test(test_data_transforms, './outputs', args.PATH_TO_CHECKPOINT, args.PATH_TO_TEST_SET, model_args)
    
def eval(model, loader, criterion, mode):
    with torch.no_grad():
        model.eval()
        N = 0
        tot_loss = 0
        l = np.array([])
        o = np.array([])
        for i, (images, labels) in enumerate(tqdm(loader)):
            #if (i+1) * batch_size > 100000:   # 100000 data samples should be sufficient to estimate the loss and the f1 score
            #    break                         # to speed up the training
            if use_gpu: 
                images, labels = images.cuda(), labels.cuda().squeeze(-1)
            outputs = model(images)

            # Loss
            loss = criterion(outputs, labels)
            N += images.shape[0]
            tot_loss += images.shape[0] * loss.item()

            # F1 score
            outputs = torch.Tensor.cpu(outputs)
            labels = torch.Tensor.cpu(labels)
            if mode == 'binning':
                outputs = outputs.sigmoid().sum(1).detach().round()
                labels = labels.sum(1)
            else:
                outputs = torch.argmax(outputs, dim=1)
            o = np.append(o, outputs)
            l = np.append(l, labels)
            
            
        f1 = f1_score(l, o, average="macro")
        accuracy = accuracy_score(l,o)
        kappa = cohen_kappa_score(l, o, weights='quadratic')
        return tot_loss/N, f1, accuracy, kappa
def eval_regression(model, loader, criterion, mode, optimized_rounder, optimize_rounder=True):
    with torch.no_grad():
        model.eval()
        N = 0
        tot_loss = 0
        preds, valid_labels = [], []
        for i, (images, labels) in enumerate(loader):
            #if (i+1) * batch_size > 100000:   # 100000 data samples should be sufficient to estimate the loss and the f1 score
            #    break                         # to speed up the training
            if use_gpu: 
                images, labels = images.cuda(), labels.cuda().squeeze(-1)
            outputs = model(images)

            # Loss
            loss = criterion(outputs, labels)
            N += images.shape[0]
            tot_loss += images.shape[0] * loss.item()
            if mode=='hybrid':
                preds.append(outputs[0].cpu().numpy())
                valid_labels.append(labels[:,:1].cpu().numpy())
            elif mode == 'both':
                preds.append(outputs[0].cpu().numpy())
                valid_labels.append(labels.cpu().numpy())
            else:
                preds.append(outputs.cpu().numpy())
                valid_labels.append(labels.cpu().numpy())
        preds = np.concatenate(preds)
        valid_labels = np.concatenate(valid_labels)
        if optimize_rounder:
            optimized_rounder.fit(preds, valid_labels)
        coefficients = optimized_rounder.coefficients()
        print(coefficients)
        final_preds = optimized_rounder.predict(preds, coefficients)
        f1 = f1_score(valid_labels, final_preds, average="macro")
        accuracy = accuracy_score(valid_labels,final_preds)
        kappa = cohen_kappa_score(valid_labels, final_preds, weights='quadratic')
        return tot_loss/N, f1, accuracy, kappa, optimized_rounder
            
def train(nepochs, batch_size, labels, path, resume_from=None, fold=0, mode='classification'):
    # Initialize logger
    experiment_time = str(datetime.datetime.now())
    logger = Logger(experiment_time)

    #SEED = 43
    #seed_everything(SEED)

    # Check device
    if use_gpu:
        logger.print_and_write('Using GPU \n')
    else:
        logger.print_and_write('Using CPU \n')
    train_data_transforms = get_aug(p=0.5, train=True)
    validation_data_transforms = get_aug(train=False)

    # Load data
    num_workers = 4
    logger.write("the train transforms are \n {}\n".format(train_data_transforms))
    logger.write("batch size is {} \n".format(batch_size))
    print('started data loading ...')
        
    N=16
    df = pd.read_csv(labels)
    print(df.columns)
    train_dataset = CustomDataset(df,N, path, fold, train=True, transforms=train_data_transforms, mode=mode)
    valid_dataset = CustomDataset(df,N, path, fold, train=False, transforms=validation_data_transforms, mode=mode)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)
    logger.print_and_write(f"number of training imgs is   {len(train_dataset)} in {len(train_loader)} batches")
    logger.print_and_write(f"number of validation imgs is {len(valid_dataset)} in {len(valid_loader)} batches \n")
    print('finished data loading !')

    # Initialize a model according to the name of model defined in params.py
    model = Binning_Attention_Model() #effnet(mode=mode)#MyResNet34(mode=mode)#effnet()# Model(mode=mode)
    if use_gpu: model.cuda()
    logger.write(f'{model} \n')
    
    if mode =='classification': criterion = nn.CrossEntropyLoss()
    elif mode =='regression': criterion = mse_loss
    elif mode =='binning': criterion = nn.BCEWithLogitsLoss()
    elif mode == 'hybrid': criterion = hybrid_loss
    else: criterion = both_loss
    
    if mode != 'classification': 
        optimized_rounder = OptimizedRounder()
    else: optimized_rounder = None
    
    #criterion = nn.CrossEntropyLoss(weight=weights.cuda())
    #optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)
    if resume_from is not None:
        print('Loading from checkpoint ...')
        checkpoint = torch.load(resume_from)
        try:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

        except:
            model.load_state_dict(checkpoint)

    models_dir = os.path.join(logger.logdir, 'models')
    os.makedirs(models_dir)

    #best_loss = 1e10
    best_kappa = 0
    #total_train_loss = 0
    for epoch in range(nepochs):

        logger.print_and_write(f'STARTING EPOCH {epoch}')
        model.train()   
        bar = tqdm(train_loader)
        train_loss = []
        for batch_idx, (images, labels) in enumerate(bar):
            if use_gpu: 
                images, labels = images.cuda(), labels.cuda().squeeze(-1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            #loss = criterion(outputs[0].float(), labels[:,0].float())
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().cpu().numpy())
            smooth_loss = sum(train_loss[-10:]) / min(len(train_loss), 10)
            bar.set_description('loss: %.5f' % (smooth_loss))
            
        ## compute train and val losses 
        if mode in ['classification', 'binning']:
            val_loss, val_f1, val_accuracy, val_kappa = eval(model, valid_loader, criterion, mode)
            train_loss, train_f1, train_accuracy, train_kappa = eval(model, train_loader, criterion, mode)
            
        else:
            val_loss, val_f1, val_accuracy, val_kappa, optimized_rounder = eval_regression(
                model, valid_loader, criterion, mode, optimized_rounder, optimize_rounder=True)

            train_loss, train_f1, train_accuracy, train_kappa, optimized_rounder = eval_regression(
                model, train_loader, criterion, mode, optimized_rounder, optimize_rounder=False)

        logger.print_and_write('Epoch %d train | loss: %.3f - accuracy: %.3f - f1 score: %.3f - kappa: %.3f'\
            %(epoch, train_loss, train_accuracy, train_f1, train_kappa))
        logger.print_and_write('Epoch %d valid | loss: %.3f - accuracy: %.3f - f1 score: %.3f - kappa: %.3f'\
            %(epoch, val_loss, val_accuracy, val_f1, val_kappa))
        #total_train_loss = 0
        ## scheduler step
        scheduler.step(val_loss)
        
        ## save a model every time the validation loss improves
        # if val_loss < best_loss: 
        #     logger.write('saving new best model !')
        #     model_file = os.path.join(models_dir, f'{model_name}_best.pth')
        #     torch.save(model.state_dict(), model_file)
        #     best_loss = val_loss

        ## save a model every time the f1 score improves
        if val_kappa > best_kappa: 
            logger.print_and_write('saving new best model !')
            model_file = os.path.join(models_dir, 'best.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'rounder': optimized_rounder.coefficients() if mode not in ['classification', 'binning'] else None,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(checkpoint, model_file)
            best_kappa = val_kappa
    

    logger.print_and_write(f"Finished training at {datetime.datetime.now()} \n")
    logger.print_and_write("Starting prediction ... \n")


    #test(test_data_transforms, logger.logdir, os.path.join(models_dir, f'{model_name}_best.pth'), test_datadir, model_args)

    #logger.print_and_write(f"Experiment finished at {datetime.datetime.now()}")

if __name__ == "__main__":
    main()