# EECS 545 Fall 2020
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

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

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


def _train_epoch(train_loader, model, criterion, optimizer):
    """
    Train the model for one iteration through the train set.
    """
    y_true, y_pred = [], []
    print("start train this epoch")
    for i, (X, y) in enumerate(train_loader):
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        y=y.to(device)
        output = model(X.to(device))
        predicted = predictions(output.data)
        y_true.append(y)
        y_pred.append(predicted)
        loss = criterion(output, y).to(device)
        loss.backward()
        optimizer.step()
        if((i%int(len(train_loader)/5))==0):
            print(format(i/len(train_loader)*100, '.2f'),'% complete, loss: ', loss.item())
        #if((i%int(len(train_loader)/5))==0):
        #    break
    
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    total_acc = accuracy(y_true, y_pred)
    print('training accuracy: ', format(total_acc, '.2f'))
    
    


def _evaluate_epoch(train_loader, val_loader, model, criterion, epoch):
    """
    Evaluates the model on the train and validation set.
    """
    print("start evaluate this epoch")
    stat = []
    i=0
    for data_loader in [val_loader]:
        y_true, y_pred, running_loss = evaluate_loop(data_loader, model, criterion)
        total_loss = np.sum(running_loss) / y_true.size(0)
        total_acc = accuracy(y_true, y_pred)
        stat += [total_acc, total_loss]
        i += 1
        
        
    print('evaluate accuracy: ', format(total_acc, '.2f'))
    #plotter.stats.append(stat)
    #plotter.log_cnn_training(epoch)
    #plotter.update_cnn_training_plot(epoch)


def evaluate_loop(data_loader, model, criterion=None):
    model.eval()
    y_true, y_pred, running_loss = [], [], []
    for i,(X, y) in enumerate(data_loader):
        X=X.to(device)
        y=y.to(device)
        
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            if criterion is not None:
                running_loss.append(criterion(output, y).item() * X.size(0))
        if((i%(int(len(data_loader)/5)))==0):
            print(format(i/len(data_loader)*100, '.2f'))
        #if((i%int(len(data_loader)/5))==0):
        #    break
    model.train()
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    return y_true, y_pred, running_loss


def train(config, dataset, model):
    # Data loaders
    train_loader, val_loader = dataset.train_loader, dataset.val_loader

    if 'use_weighted' not in config:
        # TODO (part c): define loss function
        criterion = torch.nn.CrossEntropyLoss()
    else:
        # TODO (part e): define weighted loss function
        criterion = torch.nn.CrossEntropyLoss(weight = torch.Tensor([1,2,5]).to(device))
    # TODO (part c): define optimizer
    learning_rate = config['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-6)
    #optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, weight_decay=1e-6)
    # Attempts to restore the latest checkpoint if exists
    print('Loading model...')
    #force = config['ckpt_force'] if 'ckpt_force' in config else False
    #model, start_epoch, stats = checkpoint.restore_checkpoint(model, config['ckpt_path'], force=force)
    #checkpoint = torch.load(config['ckpt_path'])
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #start_epoch = checkpoint['epoch']+1
    start_epoch=0

    # Create plotter
    #plot_name = config['plot_name'] if 'plot_name' in config else 'CNN'
    #plotter = Plotter(stats, plot_name)

    # Evaluate the model
    #_evaluate_epoch(train_loader, val_loader, model, criterion, start_epoch)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config['num_epoch']):
        print('epoch: ', epoch)
        # Train model on training set
        _train_epoch(train_loader, model, criterion, optimizer)

        # Evaluate model on training and validation set
        _evaluate_epoch(train_loader, val_loader, model, criterion, epoch + 1)

        # Save model parameters
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'loss': LOSS,
            }, config['ckpt_path']+'_'+str(epoch))
        #checkpoint.save_checkpoint(model, epoch + 1, config['ckpt_path'], plotter.stats)

    print('Finished Training')

    # Save figure and keep plot open
    #plotter.save_cnn_training_plot()
    #plotter.hold_training_plot()
    
def test(config, path, model):
    checkpoint = torch.load(config['ckpt_path']+'_'+str(3))
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
        'ckpt_path': 'checkpoints/cnnmodel.pt',  # directory to save our model checkpoints
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
    train(config, dataset, model)
    
    #test(config, 'classes/test', model)
