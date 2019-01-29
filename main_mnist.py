
# coding: utf-8

# In[19]:

import data
import util

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter

import os
import argparse
import numpy as np
from glob import glob
from collections import deque
from tqdm import tqdm

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu_number', type=str, default='0')
# parser.add_argument('--logdir',     type=str, required=True)
parser.add_argument('--epochs',     type=int, default=100)
parser.add_argument('--lr',         type=float, default=1e-2)
parser.add_argument('--bs',         type=int, default=200)
parser.add_argument('--wu',         type=int, default=300)
parser.add_argument('--threshold',  type=float, default=0.333)
parser.add_argument('--acc_snap',   type=int, default=100)
parser.add_argument('--noise_pow',  type=float, default=0, help='lr noise pow for our algorithm')
args = parser.parse_args()

# setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
channels = [1,32,64,64]
fc_size = 512
num_classes = 10
num_epochs = args.epochs
batch_size = args.bs
lr = args.lr
warm_up = args.wu
threshold = args.threshold
acc_snap = args.acc_snap


logdir = 'lr{}_bs{}_th{}_pow{}'.format(args.lr,args.bs,args.threshold,args.noise_pow)
# Directory
parent_dir = os.path.join('runs',logdir)
log_dir = os.path.join(parent_dir, 'log')
ckpt_dir = os.path.join(parent_dir, 'ckpt')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


# In[21]:

train_loader, test_loader = data.mnist_dataset(batch_size)



# In[22]:


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(channels[0], channels[1], 5, padding=2)
        self.conv2 = nn.Conv2d(channels[1], channels[2], 5, padding=2)
        self.conv3 = nn.Conv2d(channels[2], channels[3], 5, padding=2)
        self.fc1 = nn.Linear(channels[3]*4*4, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, channels[3]*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[23]:


model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

total_step = len(train_loader)
print(model)


class Gradient():
    def __init__(self, params):
        '''
        params should be list format with each index in torch Tensor format
        for example, layers = list(model.layer1.parameters())
        then it would be weight, bias, weight, bias, ... parameters
        '''
        # self.params_grad.shape = batch * (number of parameters)
        n_params = 0
        for param in params:
            n_params += np.prod(param.shape)
        self.params_grad = torch.zeros([batch_size, n_params], dtype=torch.float32).to(device)
    
    def backward(self, params, index):
        # broadcasting
        start = 0
        for param in params:
            n_param = np.prod(param.shape)
            self.params_grad[index, start:start+n_param] = param.grad.view(-1)
            start += n_param
        
    def uncertainty(self):
        denom = self.params_grad.pow(2).sum()
        numer = self.params_grad.sum(dim=0).pow(2).sum() - denom
        uncer = - numer / (denom+1e-100)
            
        return uncer, numer, denom
        


# In[25]:


writer = SummaryWriter(log_dir)

def write_loss_scalar(itr, loss):
    writer.add_scalar('logs/loss', loss, itr)
    
def write_test_scalar(itr, correct):
    writer.add_scalar('logs/correct', correct, itr)
    
def write_uncertainty_scalar(itr, unc, numer, denom):
    writer.add_scalar('logs/uncertainty', unc, itr)
    writer.add_scalars('logs/numer_denom',{'numerator':numer,
                                           'denominator':denom},itr)


# In[8]:


def test():
    correct = 0.
    for (images, labels) in test_loader:
        padds = torch.zeros([batch_size, 1, 32,32])
        padds[:,:,2:30,2:30] = images
        images, labels = padds.to(device), labels.to(device)
        
        # Forward
        output = model(images)
        pred = output.max(dim=1)[1]
        correct += pred.eq(labels).cpu().sum().item()
        
    return correct/(len(test_loader)*batch_size)


# In[18]:


# initialize
sampling = 0
params = list(model.parameters())
params_grad = Gradient(params)

model.train()
for epoch in range(num_epochs):
    print('epoch {}'.format(epoch+1))
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        iteration = i+total_step*epoch
        
        padds = torch.zeros([batch_size, 1, 32,32])
        padds[:,:,2:30,2:30] = images
        images, labels = padds.to(device), labels.to(device)

        # 크기 100인 batch 안에서 weight마다 100개의 gradient에 대한 uncertainty
        for j in range(batch_size):             
            sample_image = images[j:j+1,:]
            sample_label = labels[j:j+1]

            sample_output = model(sample_image)
            optimizer.zero_grad()
            loss = criterion(sample_output, sample_label)
            loss.backward()

            params_grad.backward(params, j)
            
        uncertainty, numer, denom = params_grad.uncertainty()      
        
        # model update
        output = model(images)
        optimizer.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        # Langevin noise
        if iteration > warm_up and uncertainty > threshold:
            sampling += 1
            torch.save(model.state_dict(), os.path.join(ckpt_dir,str(iteration)+'.pth'))
            print('iter: {}, accuracy: {}'.format(iteration, test()))
            for p in model.parameters():
                p.grad += (torch.randn(p.grad.shape)).to(device) * torch.pow(torch.tensor(lr),args.noise_pow)
            
        optimizer.step()
        
        write_loss_scalar(iteration, loss)
        write_uncertainty_scalar(iteration, uncertainty, numer, denom)
                                        
        if sampling == 60:
            break
        
        if (i+1) % acc_snap == 0:
            correct = test()
            write_test_scalar(iteration, correct)
    
    if sampling == 60:
        break
