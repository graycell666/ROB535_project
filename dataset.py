import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.utils.data as data
from PIL import Image

#torch.manual_seed(1)


class Dataset:
    def __init__(self, batch_size=4, dataset_path='classes'):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.train_set = self.get_train_numpy()
        #self.x_mean, self.x_std = self.compute_train_statistics()
        self.transform, self.transform1 = self.get_transforms()
        self.train_loader, self.val_loader = self.get_dataloaders()

    def get_train_numpy(self):
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'train'))
        # Random split
        #train_set_size = int(len(train_dataset) * 0.8)
        #valid_set_size = len(train_dataset) - train_set_size
        #train_set, valid_set = data.random_split(train_dataset, [train_set_size, valid_set_size])
        #train_x = np.zeros((len(train_set), 1052, 1914, 3))
        #for i, (img, _) in enumerate(train_set):
        #    train_x[i] = img
        #return train_x / 255.0
        return train_dataset

    def compute_train_statistics(self):
        # TODO (part a): compute per-channel mean and std with respect to self.train_dataset
        x_mean = np.zeros((3))
        train_x = np.zeros((1052, 1914, 3))
        for i, (img, _) in enumerate(self.train_set):
            train_x = np.array(img)
            train_x = train_x / 255.0
            
            x_mean[0] += np.sum(train_x[:,:,0])/(1052.0*1914)
            x_mean[1] += np.sum(train_x[:,:,1])/(1052.0*1914)
            x_mean[2] += np.sum(train_x[:,:,2])/(1052.0*1914)
            
        x_mean[0] = x_mean[0]/len(self.train_set);
        x_mean[1] = x_mean[1]/len(self.train_set);
        x_mean[2] = x_mean[2]/len(self.train_set);
          # per-channel mean
        #x_mean[0] = np.mean(self.train_dataset[:,:,:,0])
        #x_mean[1] = np.mean(self.train_dataset[:,:,:,1])
        #x_mean[2] = np.mean(self.train_dataset[:,:,:,2])
        x_std = np.zeros((3))  # per-channel std
        
        for i, (img, _) in enumerate(self.train_set):
            train_x = np.array(img)
            train_x = train_x / 255.0
            n1=0;
            n2=0;
            n3=0;
            for j in range(1052):
                for k in range(1914):
                    n1 += (train_x[j,k,0]-x_mean[0])**2
                    n2 += (train_x[j,k,1]-x_mean[1])**2
                    n3 += (train_x[j,k,2]-x_mean[2])**2
            x_std[0] += n1/(1052.0*1914)
            x_std[1] += n2/(1052.0*1914)
            x_std[2] += n3/(1052.0*1914)
        x_std[0] = np.sqrt(x_std[0]/len(self.train_set))
        x_std[1] = np.sqrt(x_std[1]/len(self.train_set))
        x_std[2] = np.sqrt(x_std[2]/len(self.train_set))
        #x_std[0] = np.std(self.train_dataset[:,:,:,0])
        #x_std[1] = np.std(self.train_dataset[:,:,:,1])
        #x_std[2] = np.std(self.train_dataset[:,:,:,2])
        print(x_mean,x_std)
        return x_mean, x_std

    def get_transforms(self):
        # TODO (part a): fill in the data transforms
        transform_list = [
            # resize the image to 32x32x3
            #transforms.RandomResizedCrop((256,256), scale=(0.8,1.0)),
            transforms.Resize((1024,1024)),
            #transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(),
            # convert image to PyTorch tensor
            transforms.ToTensor(),
            # normalize the image (use self.x_mean and self.x_std)
            transforms.Normalize([0.36205036,0.35747677,0.34707443], [0.26593808,0.25821175,0.24841977], inplace=False)
        ]
        transform_list1 = [
            # resize the image to 32x32x3
            #transforms.RandomResizedCrop((256,256), scale=(0.8,1.0)),
            transforms.Resize((1024,1024)),
            # convert image to PyTorch tensor
            transforms.ToTensor(),
            # normalize the image (use self.x_mean and self.x_std)
            transforms.Normalize([0.36205036,0.35747677,0.34707443], [0.26593808,0.25821175,0.24841977], inplace=False)
        ]
        transform = transforms.Compose(transform_list)
        transform1 = transforms.Compose(transform_list1)
        return transform, transform1

    def get_dataloaders(self):
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'train'), transform=self.transform)
        val_dataset = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'val'), transform=self.transform1)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # validation set
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def plot_image(self, image, label):
        image = np.transpose(image.numpy(), (1, 2, 0))
        image = image * np.array([0.26593808,0.25821175,0.24841977]).reshape(1, 1, 3) + np.array([0.36205036,0.35747677,0.34707443]).reshape(1, 1, 3)  # un-normalize
        plt.title(label)
        plt.imshow(image)
        plt.show()

if __name__ == '__main__':
    dataset = Dataset()
    #print('x_mean: ',dataset.x_mean, '\nx_std: ',dataset.x_std)
    images, labels = iter(dataset.train_loader).next()
    dataset.plot_image(
        torchvision.utils.make_grid(images),
        ', '.join(str(labels))
    )
