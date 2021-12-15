# EECS 545 Fall 2020
import math
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network.
    """
    def __init__(self):
        super().__init__()

        # TODO (part b): define layers
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)  # convolutional layer 1
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)  # convolutional layer 2
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=2)  # convolutional layer 3
        self.conv4 = nn.Conv2d(64, 128, 5, stride=2, padding=2)  # convolutional layer 4
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32768,64)  # fully connected layer 1
        self.fc2 = nn.Linear(64, 3)  # fully connected layer 2 (output layer)

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / math.sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        # TODO (part b): initialize parameters for fully connected layers
        for fc in [self.fc1, self.fc2]:
            nn.init.normal_(fc.weight, 0.0, 1 / math.sqrt(fc.weight.size(1)))
            nn.init.constant_(conv.bias, 0.0)


    def forward(self, x):
        N, C, H, W = x.shape
        

        # TODO (part b): forward pass of image through the network
        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))
        z = self.flatten(z)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        
        z = F.log_softmax(z, dim=1)

        return z


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




if __name__ == '__main__':
    from dataset import Dataset
    net = CNN()
    print(net)
    print('Number of CNN parameters: {}'.format(count_parameters(net)))
    dataset = Dataset()
    images, labels = iter(dataset.train_loader).next()
    print('Size of model output:', net(images).size())
