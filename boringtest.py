#Taken from: https://github.com/yunjey/pytorch-tutorial

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from fast_soft_sort.pytorch_ops import soft_rank, soft_sort
from fast_soft_sort.numpy_ops import rank, sort
from pytorch_soft_sort import soft_sort_pytorch
import sys
from tqdm import tqdm
from entmax import entmax15, sparsemax
from sinkhorn.topk import TopK_stablized, TopK_custom
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 1024
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.CIFAR10(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()

        self.tks0 = TopK_custom(600, max_iter=50)
        self.tks1 = TopK_custom(400, max_iter=50)
        self.tks2 = TopK_custom(400, max_iter=50)
        self.tks3 = TopK_custom(400, max_iter=50)

        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size) 
        self.fc4 = nn.Linear(hidden_size, num_classes)

    def sort_back_to_vec(self, inp):
        #zeros vector
        zrs = torch.zeros((batch_size, 200)).cuda()
        #Get the descending indexes
        dsc_indx = soft_sort(inp.view(batch_size, -1).cpu(), "ASCENDING").cuda()
        _, indices = torch.sort(inp, descending=True)
        dsc_indx = dsc_indx.narrow(-1, 100, 400)
        
        #Scatter add back to the original array such that we have zeros everywhere else
        #zrs.scatter_add_(-1, indices.long(), dsc_indx.float()).cuda()
        return dsc_indx.float()
    def forward(self, x):
        
        sparse = int(float(random.randint(1,10))/10.)

        out = self.fc1(x)
        out = self.relu(out)
        out = out * sparse * self.tks1(out) + (1-sparse) * out

        out = self.fc2(out)
        out = self.relu(out)
        out = out * sparse * self.tks2(out) + (1-sparse) * out

        out = self.fc3(out)
        out = self.relu(out)
        out = out * sparse * self.tks3(out) + (1-sparse) * out

        out = self.fc4(out)
        
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)
for epoch in tqdm(range(num_epochs)):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 32*32).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 32*32).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
