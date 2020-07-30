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
num_epochs = 40
batch_size = 100
learning_rate = 0.05
n = 2
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
        self.fc1 = nn.Linear(input_size*3, hidden_size) 
        self.relu = nn.ReLU()

        #self.tks0 = TopK_custom(600, max_iter=50)
        #self.tks1 = TopK_custom(400, max_iter=50)
        self.tks3 = TopK_custom(400, max_iter=50)
        self.tks4 = TopK_custom(400, max_iter=50)

        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size) 
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, num_classes)

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
        
        sparse = int(float(random.randint(1,n))/float(n))

        out = self.fc1(x)
        out = self.relu(out)
        out = out #* sparse * self.tks1(out) + (1-sparse) * out

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)
        #out = out * sparse * self.tks3(out) + (1-sparse) * out

        out = self.fc4(out)
        out = self.relu(out)
        #out = out * sparse * self.tks4(out) + (1-sparse) * out
        
        return self.fc5(out)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 200)
        self.fc2 = torch.nn.Linear(200, 84)
        self.fc3 = torch.nn.Linear(84, 10)

        self.tks1 = TopK_custom(120, max_iter=100)
        self.tks2 = TopK_custom(75, max_iter=50)
        self.relu = nn.ReLU()

    def forward(self, x, test = False):
        sparse = int(float(random.randint(1,n))/float(n))
        if test:
            sparse = 0


        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = sparse * self.tks1(x) + (1-sparse) * x

        x = self.relu(self.fc2(x))
        #x = sparse * self.tks2(x) + (1-sparse) * x

        x = self.fc3(x)
        return x


net = Net()
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
model = Net().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)

def test():
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, True)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

for epoch in tqdm(range(num_epochs)):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
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

    test()


# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
