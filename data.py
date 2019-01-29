import torch
import torchvision
import torchvision.transforms as transforms
    
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from scipy.misc import imread
from torch import Tensor

def mnist_dataset(batch_size=100):
    # MNIST dataset 
    # transform_train = transforms.Compose([transforms.RandomCrop(28,padding=2),
    #                                       transforms.RandomHorizontalFlip(),
    #                                       transforms.ToTensor(),
    #                                      ])
    
    train_dataset = torchvision.datasets.MNIST(root='data/mnist', 
                                               train=True, 
                                               transform=transforms.ToTensor(),  
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/mnist', 
                                              train=False, 
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)
    
    return train_loader, test_loader

def fashion_mnist_dataset(batch_size=100):
    train_dataset = torchvision.datasets.FashionMNIST(root='data/f_mnist', 
                                               train=True, 
                                               transform=transforms.ToTensor(),  
                                               download=True)

    test_dataset = torchvision.datasets.FashionMNIST(root='data/f_mnist', 
                                              train=False, 
                                              transform=transforms.ToTensor())
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)
    
    return train_loader, test_loader

def not_mnist_dataset(batch_size=100):
    train_dataset = notMNIST(os.path.join('data','not_mnist','Train'))
    test_dataset = notMNIST(os.path.join('data','not_mnist','Test'))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    test_loader =torch.utils.data. DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)
    
    return train_loader, test_loader

"""
Code from https://github.com/Aftaab99/PyTorchImageClassifier

Loads the train/test set. 
Every image in the dataset is 28x28 pixels and the labels are numbered from 0-9
for A-J respectively.

Set root to point to the Train/Test folders.
"""

# Creating a sub class of torch.utils.data.dataset.Dataset
class notMNIST(Dataset):

	# The init method is called when this class will be instantiated.
	def __init__(self, root):
		Images, Y = [], []
		folders = os.listdir(root)

		for folder in folders:
			folder_path = os.path.join(root, folder)
			for ims in os.listdir(folder_path):
				try:
					img_path = os.path.join(folder_path, ims)
					Images.append(np.array(imread(img_path)))
					Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
				except:
					# Some images in the dataset are damaged
					print("File {}/{} is broken".format(folder, ims))
		data = [(x, y) for x, y in zip(Images, Y)]
		self.data = data

	# The number of items in the dataset
	def __len__(self):
		return len(self.data)

	# The Dataloader is a generator that repeatedly calls the getitem method.
	# getitem is supposed to return (X, Y) for the specified index.
	def __getitem__(self, index):
		img = self.data[index][0]

		# 8 bit images. Scale between [0,1]. This helps speed up our training
		img = img.reshape(28, 28) / 255.0

		# Input for Conv2D should be Channels x Height x Width
		img_tensor = Tensor(img).view(1, 28, 28).float()
		label = self.data[index][1]
		return (img_tensor, label)
