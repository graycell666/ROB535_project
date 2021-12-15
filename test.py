# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 20:23:00 2021

@author: 10675
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
import random
#import checkpoint
from dataset import Dataset
from model import CNN
from resnet import ResNet18
#from plot import Plotter
import csv
from PIL import Image
from glob import glob

import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predictions(logits):
    """
    Compute the predictions from the model.
    Inputs:
        - logits: output of our model based on some input, tensor with shape=(batch_size, num_classes)
    Returns:
        - pred: predictions of our model, tensor with shape=(batch_size)
    """
    # TODO (part c): compute the predictions
    prediction = torch.argmax(logits, dim=1)
    #print(prediction[0].item())

    return prediction


def accuracy(y_true, y_pred):
    """
    Compute the accuracy given true and predicted labels.
    Inputs:
        - y_true: true labels, tensor with shape=(num_examples)
        - y_pred: predicted labels, tensor with shape=(num_examples)
    Returns:
        - acc: accuracy, float
    """
    # TODO (part c): compute the accuracy
    
    acc=0
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    #print(y_true)
    #for i in range(len(y_true)):
    #    if(y_true[i]>=0 and y_true[i]<=7 and y_pred[i]>=0 and y_pred[i]<=7):
    #        acc += 1
    #    elif(y_true[i]>=8 and y_true[i]<=10 and y_pred[i]>=8 and y_pred[i]<=10):
    #        acc += 1
    #    elif(y_true[i]>=11 and y_true[i]<=12 and y_pred[i]>=11 and y_pred[i]<=12):
    #        acc += 1
    miss2 = 0
    miss1 = 0
    num1 = 0
    num2 = 0
    for i in range(len(y_true)):
        if(y_true[i]==2):
            num2 += 1
        elif(y_true[i]==1):
            num1 += 1
    for i in range(len(y_true)):
        if(y_true[i]==2 and y_pred[i]!=2):
            miss2 += 1
        elif(y_true[i]==1 and y_pred[i]!=1):
            miss1 += 1
    print("label1 wrong:", format(miss1/num1, '.2f'))
    print("label2 wrong:", format(miss2/num2, '.2f'))
    acc = (y_true == y_pred).sum() / y_true.size(0)

    return acc

def test(config, path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    epoch = checkpoint['epoch']
    print(epoch)
    
    model.eval()
    
    files = glob('classes/test/0/*.jpg')
    files.sort()
    
    transform_list = [
        # resize the image to 32x32x3
        transforms.Resize((1024,1024)),
        # convert image to PyTorch tensor
        transforms.ToTensor(),
        # normalize the image (use self.x_mean and self.x_std)
        transforms.Normalize([0.36205036,0.35747677,0.34707443], [0.26593808,0.25821175,0.24841977], inplace=False)
    ]
    transform = transforms.Compose(transform_list)
    
    name = 'test_labels.csv'
    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['guid/image', 'label'])
        for file in files:
            img = Image.open(file)
            X = transform(img).to(device)
            #print(X.size())
            Xin = torch.zeros(1,X.size()[0],X.size()[1],X.size()[2])
            Xin[0,:,:,:] = X
            #X[0,:,:,:] = transforms.ToTensor(data).permute(2, 0, 1)
            model.to(device)
            output = model(Xin.to(device))
            predicted = predictions(output.data)
            
            #result=0
            #if(predicted>=0 and predicted<=7):
            #    result=1
            #elif(predicted>=8 and predicted<=10):
            #    result=2
            #elif(predicted>=11 and predicted<=12):
            #    result=0
            
            guid_idx = file.split('\\')[-1]
            row = guid_idx.replace('_','/').replace('.jpg','')
            writer.writerow([row, predicted.item()])
            print(row, predicted.item())

if __name__ == '__main__':
    
    # define config parameters for training
    config = {
        'dataset_path': 'classes/',
        'batch_size': 32,
        'ckpt_path': 'model.pt',  # directory to save our model checkpoints
        'num_epoch': 20,                 # number of epochs for training
        'learning_rate': 1e-3,           # learning rate
        'use_weighted': True,
    }
    # create dataset
    dataset = Dataset(config['batch_size'], config['dataset_path'])
    # create model
    #model = ResNet18().to(device)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 3)
    model = model.to(device)
    # train our model on dataset
    #train(config, dataset, model)
    
    test(config, config['ckpt_path']+'_'+str(3), model)