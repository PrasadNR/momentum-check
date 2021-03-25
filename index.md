## This is a work in progress. Autograd graph is getting messed up somehow and that needs to be fixed.

## Import all the necessary libraries

Source: https://github.com/akshat57/Blind-Descent/blob/main/Blind_Descent-1-CNN.ipynb


```python
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10

import matplotlib.pyplot as plt
import time

from sklearn.metrics import confusion_matrix, classification_report

cuda = torch.cuda.is_available()
cuda = False
```

## Download the MNIST and CIFAR10 datasets


```python
train = MNIST('./MNIST_data', train=True, download=True, transform=transforms.ToTensor())
test = MNIST('./MNIST_data', train=False, download=True, transform=transforms.ToTensor())
train_MNIST_data = train.data; train_MNIST_labels = train.targets
test_MNIST_data = test.data; test_MNIST_labels = test.targets

train = CIFAR10('./CIFAR10_data', train=True, download=True, transform=transforms.ToTensor())
test = CIFAR10('./CIFAR10_data', train=False, download=True, transform=transforms.ToTensor())
train_CIFAR10_data = train.data; train_CIFAR10_labels = train.targets
test_CIFAR10_data = test.data; test_CIFAR10_labels = test.targets

print()
print("MNIST is already an array")
print(train_MNIST_data.shape, train_MNIST_labels.shape, test_MNIST_data.shape, test_MNIST_labels.shape)
print()
print("CIFAR10 is a list of arrays")
print(len(train_CIFAR10_data), len(train_CIFAR10_labels), len(test_CIFAR10_data), len(test_CIFAR10_labels))
print(train_CIFAR10_data[0].shape, test_CIFAR10_data[0].shape)
```

    Files already downloaded and verified
    Files already downloaded and verified
    
    MNIST is already an array
    torch.Size([60000, 28, 28]) torch.Size([60000]) torch.Size([10000, 28, 28]) torch.Size([10000])
    
    CIFAR10 is a list of arrays
    50000 50000 10000 10000
    (32, 32, 3) (32, 32, 3)
    

## Dataloader


```python
class CIFAR10Dataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self,index):
        X = np.transpose(self.X[index], (2, 0, 1)) / 255
        X = X.astype(float)
        Y = self.Y[index]
        return X,Y

class MNIST_Dataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self,index):
        X = np.pad(self.X[index], 2) / 255
        X = np.repeat(X[:, :, np.newaxis], 3, axis = 2)
        X = np.transpose(X, (2, 0, 1))
        X = X.astype(float)
        Y = self.Y[index]
        return X,Y
```

Using the torch.utils.data DataLoader, we shuffle the data and set the batch size


```python
num_workers = 8 if cuda else 0 
batch_size = 256
    
# MNIST Training
train_dataset = MNIST_Dataset(train_MNIST_data, train_MNIST_labels)

train_loader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=True, batch_size=batch_size)
train_MNIST_loader = data.DataLoader(train_dataset, **train_loader_args)

# MNIST Testing
test_dataset = MNIST_Dataset(test_MNIST_data, test_MNIST_labels)

test_loader_args = dict(shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=False, batch_size=1)
test_MNIST_loader = data.DataLoader(test_dataset, **test_loader_args)

# CIFAR10 Training
train_dataset = CIFAR10Dataset(train_CIFAR10_data, train_CIFAR10_labels)

train_loader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=True, batch_size=batch_size)
train_CIFAR10_loader = data.DataLoader(train_dataset, **train_loader_args)

# CIFAR10 Testing
test_dataset = CIFAR10Dataset(test_CIFAR10_data, test_CIFAR10_labels)

test_loader_args = dict(shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=False, batch_size=1)
test_CIFAR10_loader = data.DataLoader(test_dataset, **test_loader_args)
```

## Define our Neural Network Model 
We define our model using the torch.nn.Module class


```python
class MyCNN_Model(nn.Module):
    def __init__(self):
        super(MyCNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        self.conv3 = nn.Conv2d(32, 10, kernel_size = 5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        
        return x
```

## Create the model and define the Loss and Optimizer


```python
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if cuda else "cpu")
model = MyCNN_Model()
model.to(device)
print(model)
```

    MyCNN_Model(
      (conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))
      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1))
      (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv3): Conv2d(32, 10, kernel_size=(5, 5), stride=(1, 1))
    )
    

### This train_epoch is the most important function. The idea is to try not to lose momentum just because of some abrupt local minimum (when the loss surface is not smooth). The idea is to make sure that we check two things:
### 1. The normal loss optimisation
### 2. Uniform noise added to gradients
### For each batch (in any given epoch), we pick the one of the two that produces lower loss values.
### The momentum of the optimiser though is maintained as per the gradients picked: Either the raw gradients or the gradients added with noise (as we do not want to lose momentum just because of abrupt local minimum)


```python
def train_epoch(eta, model, train_loader, criterion):
    model.train()

    running_loss = 0.0
    predictions = []
    ground_truth = []
    loss_den = 1
    
    start_time = time.time()
    optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data = data.to(device)
        target = target.to(device)
    
        #previous model
        outputs = model(data.float())
        _, predicted = torch.max(outputs.data, 1)
        total_predictions = target.size(0)
        correct_predictions = (predicted == target).sum().item()
        acc = (correct_predictions/total_predictions)*100.0
        
        loss = criterion(outputs, target)
        loss.backward()
        optimiser.step()
        
        #convGrad is the set of old gradients
        conv1grad = model.conv1.weight.grad
        conv2grad = model.conv2.weight.grad
        conv3grad = model.conv3.weight.grad
                
        noisyGrad1 = eta * np.abs(conv1grad.detach().cpu().numpy())
        noisyGrad2 = eta * np.abs(conv2grad.detach().cpu().numpy())
        noisyGrad3 = eta * np.abs(conv3grad.detach().cpu().numpy())
        
        newGrad1 = conv1grad + torch.from_numpy(np.random.uniform(-noisyGrad1, noisyGrad1))
        newGrad2 = conv2grad + torch.from_numpy(np.random.uniform(-noisyGrad2, noisyGrad2))
        newGrad3 = conv3grad + torch.from_numpy(np.random.uniform(-noisyGrad3, noisyGrad3))
        
        model.conv1.weight.grad = nn.Parameter(torch.from_numpy(newGrad1.detach().numpy()).float())
        model.conv2.weight.grad = nn.Parameter(torch.from_numpy(newGrad2.detach().numpy()).float())
        model.conv3.weight.grad = nn.Parameter(torch.from_numpy(newGrad3.detach().numpy()).float())
        
        #The new loss value for the new gradients is computed
        outputs = model(data.float())
        _, predicted = torch.max(outputs.data, 1)
        total_predictions = target.size(0)
        correct_predictions = (predicted == target).sum().item()
        acc_new = (correct_predictions/total_predictions)*100.0
        
        loss_new = criterion(outputs, target)
        loss_den += 1

        #calculuating confusion matrix
        predictions += list(predicted.detach().cpu().numpy())
        ground_truth += list(target.detach().cpu().numpy())

        if loss_new.item() > loss.item():
            model.conv1.weight.grad = conv1grad
            model.conv2.weight.grad = conv2grad
            model.conv3.weight.grad = conv3grad

            running_loss += loss.item()
        else:
            running_loss += loss_new.item()
        
    end_time = time.time()

    running_loss /= loss_den
    
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    
    return running_loss, model
```

## Create a function that will evaluate our network's performance on the test set


```python
def test_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        
        predictions = []
        ground_truth = []

        for batch_idx, (data, target) in enumerate(test_loader):   
            data = data.to(device)
            target = target.to(device)

            outputs = model(data.float())

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
            
            #calculuating confusion matrix
            predictions += list(predicted.detach().cpu().numpy())
            ground_truth += list(target.detach().cpu().numpy())
        
        #write_confusion_matrix('Testing', ground_truth, predictions)
        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc

```

## Train the model for N epochs
We call our training and testing functions in a loop, while keeping track of the losses and accuracy. 


```python
n_epochs = 4
eta = 0.01

model = MyCNN_Model(); model.to(device)
for i in range(n_epochs):
    train_loss, model = train_epoch(eta, model, train_MNIST_loader, criterion)
    test_loss, MNIST_test_acc = test_model(model, test_MNIST_loader, criterion)
    print('='*20)

model = MyCNN_Model(); model.to(device)
for i in range(n_epochs):
    train_loss, model = train_epoch(eta, model, train_CIFAR10_loader, criterion)
    test_loss, CIFAR10_test_acc = test_model(model, test_CIFAR10_loader, criterion)
    print('-'*20)
```

    Training Loss:  2.982338696465654 Time:  58.64115643501282 s
    Testing Loss:  2.4523444892406463
    Testing Accuracy:  10.32 %
    ====================
    Training Loss:  2.495215079541934 Time:  57.059688091278076 s
    Testing Loss:  2.6800487677812574
    Testing Accuracy:  10.32 %
    ====================
    Training Loss:  3.1331407286353032 Time:  57.16563701629639 s
    Testing Loss:  3.6217195371329782
    Testing Accuracy:  10.09 %
    ====================
    Training Loss:  5.119678274049598 Time:  57.953062534332275 s
    Testing Loss:  8.410976375210286
    Testing Accuracy:  9.8 %
    ====================
    Training Loss:  2.3521221628043856 Time:  46.376412868499756 s
    Testing Loss:  2.3259525919675825
    Testing Accuracy:  10.01 %
    --------------------
    Training Loss:  2.330018611123719 Time:  47.75367259979248 s
    Testing Loss:  2.3980944871902468
    Testing Accuracy:  10.0 %
    --------------------
    


```python

```
