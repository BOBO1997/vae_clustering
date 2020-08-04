#!/usr/bin/env python
# coding: utf-8

# # Autoencoderのモデルを定義

# In[1]:


# import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
# import os
# import pickle
# from PIL import Image
# from pprint import pprint
# import random
# import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler
# import torchvision
# from torchvision import datasets, transforms


# In[ ]:


class DenseAE(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        ## encoder layers ##
        self.linear1 = nn.Linear(self.img_size * self.img_size, self.img_size // 4 * self.img_size // 4)
        
        ## decoder layers ##
        self.linear2 = nn.Linear(self.img_size // 4 * self.img_size // 4, self.img_size * self.img_size)

    def forward(self, x, get_hidden = False):
        ## encode ## 
        x = self.linear1(x)
        if get_hidden:
            return torch.flatten(x)
        
        ## decode ##
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


# In[ ]:


class DenseAE_2dim(nn.Module):
    
    def __init__(self, img_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(img_size * img_size,  1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 2))
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128 * 128),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# # AutoEncoder: MaxPoolなし(2020.06.17)

# In[ ]:


class ConvAE(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 1 * 128 * 128, output: 8 * 64 * 64
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 8 * 64 * 64, output: 16 * 32 * 32
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 16 * 32 * 32, output: 32 * 16 * 16
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2) # conv layer, 3x3 kernels, input: 32 * 16 * 16, output: 64 * 8 * 8
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) # conv layer, 3x3 kernels, input: 64 * 8 * 8, output: 128 * 4 * 4
        self.conv6 = nn.Conv2d(128, 64, kernel_size=4) # conv layer, 4x4 kernels, input: 128 * 4 * 4, output: 64 * 1 * 1
        
        ## decoder layers ##
        self.t_conv6 = nn.ConvTranspose2d(64, 128, kernel_size=4, stride=1)
        self.t_conv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2)

    def forward(self, x, get_hidden = False):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        if get_hidden:
            return x.flatten()
        
        ## decode ##
        x = F.relu(self.t_conv6(x))
        x = F.relu(self.t_conv5(x))[:, :, :-1, :-1]
        x = F.relu(self.t_conv4(x))[:, :, :-1, :-1]
        x = F.relu(self.t_conv3(x))[:, :, 1:-2, 1:-2]
        x = F.relu(self.t_conv2(x))[:, :, 1:-2, 1:-2]
        x = self.t_conv1(x)[:, :, 1:-2, 1:-2]
        x = torch.sigmoid(x)
        return x


# In[ ]:


class ConvAE_deep(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 1 * 128 * 128, output: 8 * 64 * 64
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 8 * 64 * 64, output: 16 * 32 * 32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 16 * 32 * 32, output: 32 * 16 * 16
        self.conv4 = nn.Conv2d(64, 256, kernel_size=3, padding=1, stride=2) # conv layer, 3x3 kernels, input: 32 * 16 * 16, output: 64 * 8 * 8
        self.conv5 = nn.Conv2d(256, 1024, kernel_size=3, padding=1, stride=2) # conv layer, 3x3 kernels, input: 64 * 8 * 8, output: 128 * 4 * 4
        self.conv6 = nn.Conv2d(1024, 64, kernel_size=4) # conv layer, 4x4 kernels, input: 128 * 4 * 4, output: 64 * 1 * 1
        
        ## decoder layers ##
        self.t_conv6 = nn.ConvTranspose2d(64, 1024, kernel_size=4, stride=1)
        self.t_conv5 = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 8, kernel_size=5, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2)

    def forward(self, x, get_hidden = False):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        if get_hidden:
            return x.flatten()
        
        ## decode ##
        x = F.relu(self.t_conv6(x))
        x = F.relu(self.t_conv5(x))[:, :, :-1, :-1]
        x = F.relu(self.t_conv4(x))[:, :, :-1, :-1]
        x = F.relu(self.t_conv3(x))[:, :, 1:-2, 1:-2]
        x = F.relu(self.t_conv2(x))[:, :, 1:-2, 1:-2]
        x = self.t_conv1(x)[:, :, 1:-2, 1:-2]
        x = torch.sigmoid(x)
        return x


# In[ ]:


class ConvAE_2dim(nn.Module):
    def __init__(self, img_size, binary = False):
        super().__init__()
        self.img_size = img_size
        self.binary = binary
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 1 * 128 * 128, output: 8 * 64 * 64
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 8 * 64 * 64, output: 16 * 32 * 32
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 16 * 32 * 32, output: 32 * 16 * 16
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2) # conv layer, 3x3 kernels, input: 32 * 16 * 16, output: 64 * 8 * 8
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) # conv layer, 3x3 kernels, input: 64 * 8 * 8, output: 128 * 4 * 4
        self.conv6 = nn.Conv2d(128, 128, kernel_size=4) # conv layer, 4x4 kernels, input: 128 * 4 * 4, output: 2 * 1 * 1
        self.conv7 = nn.Conv2d(128, 2, kernel_size=1) # conv layer, 4x4 kernels, input: 128 * 4 * 4, output: 2 * 1 * 1
        # self.linear = nn.Linear(64, 2)
        
        ## decoder layers ##
        # self.t_linear = nn.Linear(2, 64)
        self.t_conv7 = nn.ConvTranspose2d(2, 128, kernel_size=1, stride=1)
        self.t_conv6 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=1)
        self.t_conv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2)
        
    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        return x
        
    def decoder(self, x):
        x = F.relu(self.t_conv7(x))
        x = F.relu(self.t_conv6(x))
        x = F.relu(self.t_conv5(x))[:, :, :-1, :-1]
        x = F.relu(self.t_conv4(x))[:, :, :-1, :-1]
        x = F.relu(self.t_conv3(x))[:, :, 1:-2, 1:-2]
        x = F.relu(self.t_conv2(x))[:, :, 1:-2, 1:-2]
        x = self.t_conv1(x)[:, :, 1:-2, 1:-2]
        x = torch.sigmoid(x) if self.binary else torch.sigmoid(x) # 要修正 # 二値化
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # x = F.relu(self.linear(x.view(-1, 64)))
        # x = F.relu(self.t_linear(x))
        # x = F.relu(self.t_conv6(x.view(-1, 64, 1, 1)))
        return x


# In[ ]:


class ConvAE_2dim_alta(nn.Module):
    def __init__(self, img_size, binary = False):
        super().__init__()
        self.img_size = img_size
        self.binary = binary
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 1 * 128 * 128, output: 8 * 64 * 64
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 8 * 64 * 64, output: 16 * 32 * 32
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 16 * 32 * 32, output: 32 * 16 * 16
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2) # conv layer, 3x3 kernels, input: 32 * 16 * 16, output: 64 * 8 * 8
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) # conv layer, 3x3 kernels, input: 64 * 8 * 8, output: 128 * 4 * 4
        self.conv6 = nn.Conv2d(128, 128, kernel_size=4) # conv layer, 4x4 kernels, input: 128 * 4 * 4, output: 2 * 1 * 1
        self.conv7 = nn.Conv2d(128, 2, kernel_size=1) # conv layer, 4x4 kernels, input: 128 * 4 * 4, output: 2 * 1 * 1
        # self.linear = nn.Linear(64, 2)
        
        ## decoder layers ##
        # self.t_linear = nn.Linear(2, 64)
        self.t_conv7 = nn.ConvTranspose2d(2, 128, kernel_size=1, stride=1)
        self.t_conv6 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=1)
        self.t_conv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2)
        
    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        return x
        
    def decoder(self, x):
        x = F.relu(self.t_conv7(x))
        x = F.relu(self.t_conv6(x))
        x = F.relu(self.t_conv5(x))[:, :, :-1, :-1]
        x = F.relu(self.t_conv4(x))[:, :, :-1, :-1]
        x = F.relu(self.t_conv3(x))[:, :, 1:-2, 1:-2]
        x = F.relu(self.t_conv2(x))[:, :, 1:-2, 1:-2]
        x = self.t_conv1(x)[:, :, 1:-2, 1:-2]
        x = torch.sigmoid(x) if self.binary else torch.sigmoid(x) # 要修正 # 二値化
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # x = F.relu(self.linear(x.view(-1, 64)))
        # x = F.relu(self.t_linear(x))
        # x = F.relu(self.t_conv6(x.view(-1, 64, 1, 1)))
        return x


# In[ ]:


class ConvAE_mnist(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1) # conv layer (channel from 1 --> 2), 3x3 kernels, input: 1 * 28 * 28,
        self.conv2 = nn.Conv2d(2, 3, 3, padding=1) # conv layer (channel from 2 --> 3), 3x3 kernels, input: 2 * 14 * 14
        self.pool = nn.MaxPool2d(2, 2) # pooling layer to reduce x-y dims by two; kernel and stride of 2
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv4 = nn.ConvTranspose2d(3, 2, 2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(2, 1, 2, stride=2)

    def forward(self, x, get_hidden = False):
        ## encode ##
        # add hidden layers with relu activation function
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        if get_hidden:
            return torch.flatten(x)
        
        ## decode ##
        x = F.relu(self.t_conv4(x)) # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv5(x)) # output layer (with sigmoid for scaling from 0 to 1)
        return x


# # ここからVAE (2020.06.24)

# In[ ]:


class DenseVAE(nn.Module):
    def __init__(self, img_size=128, z_dim=64):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear1 = nn.Linear(img_size ** 2, 4096)
        self.linear2 = nn.Linear(4096, 1024)
        
        self.linear_mean = nn.Linear(1024, z_dim)
        self.linear_logvar = nn.Linear(1024, z_dim)
        
        self.dec_linear3 = nn.Linear(z_dim, 1024)
        self.dec_linear2 = nn.Linear(1024, 4096)
        self.dec_linear1 = nn.Linear(4096, img_size ** 2)
    
    def encoder(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.linear_mean(x)
        logvar = F.softplus(self.linear_logvar(x))
        return mean, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoder(self, z):
        x = F.relu(self.dec_linear3(z))
        x = F.relu(self.dec_linear2(x))
        x = torch.sigmoid(self.dec_linear1(x))
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# In[ ]:


class DenseVAE_2dim(nn.Module):
    def __init__(self, img_size=128, z_dim=2):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear1 = nn.Linear(img_size ** 2, 4096)
        self.linear2 = nn.Linear(4096, 1024)
        self.linear3 = nn.Linear(1024, 256)
        self.linear4 = nn.Linear(256, 64)
        self.linear5 = nn.Linear(64, 16)
        
        self.linear_mean = nn.Linear(16, z_dim)
        self.linear_logvar = nn.Linear(16, z_dim)
        
        self.dec_linear6 = nn.Linear(z_dim, 16)
        self.dec_linear5 = nn.Linear(16, 64)
        self.dec_linear4 = nn.Linear(64, 256)
        self.dec_linear3 = nn.Linear(256, 1024)
        self.dec_linear2 = nn.Linear(1024, 4096)
        self.dec_linear1 = nn.Linear(4096, img_size ** 2)
    
    def encoder(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        mean = self.linear_mean(x)
        logvar = F.softplus(self.linear_logvar(x))
        return mean, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoder(self, z):
        x = F.relu(self.dec_linear6(z))
        x = F.relu(self.dec_linear5(x))
        x = F.relu(self.dec_linear4(x))
        x = F.relu(self.dec_linear3(x))
        x = F.relu(self.dec_linear2(x))
        x = torch.sigmoid(self.dec_linear1(x))
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# In[ ]:


class ConvVAE(nn.Module):
    def __init__(self, img_size=128, z_dim=64):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 1 * 128 * 128, output: 16 * 64 * 64
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 16 * 64 * 64, output: 64 * 32 * 32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 64 * 32 * 32, output: 128 * 16 * 16
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2) # conv layer, 3x3 kernels, input:128 * 16 * 16, output: 256 * 8 * 8
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2) # conv layer, 3x3 kernels, input: 256 * 8 * 8, output: 512 * 4 * 4
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=4) # conv layer, 4x4 kernels, input: 512 * 4 * 4, output: 1024 * 1 * 1
        
        self.conv_mean = nn.Conv2d(1024, z_dim, kernel_size=1)
        self.conv_var = nn.Conv2d(1024, z_dim, kernel_size=1)
        
        self.t_conv7 = nn.ConvTranspose2d(z_dim, 1024, kernel_size=1, stride=1)
        self.t_conv6 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=1)
        self.t_conv5 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(64, 16, kernel_size=5, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2)
        
    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        mean = self.conv_mean(x)
        logvar = F.softplus(self.conv_var(x))
        return mean, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoder(self, z):
        x = F.relu(self.t_conv7(z))
        x = F.relu(self.t_conv6(x))
        x = F.relu(self.t_conv5(x))[:, :, :-1, :-1]
        x = F.relu(self.t_conv4(x))[:, :, :-1, :-1]
        x = F.relu(self.t_conv3(x))[:, :, 1:-2, 1:-2]
        x = F.relu(self.t_conv2(x))[:, :, 1:-2, 1:-2]
        x = torch.sigmoid(self.t_conv1(x))[:, :, 1:-2, 1:-2]
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# In[ ]:


class ConvVAE_224(nn.Module):
    def __init__(self, img_size=128, z_dim=64):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 1 * 224 * 224, output: 16 * 112 * 112
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 16 * 112 * 112, output: 64 * 56 * 56
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2) # conv layer, 5x5 kernels, input: 64 * 56 * 56, output: 128 * 28 * 28
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2) # conv layer, 3x3 kernels, input: 128 * 28 * 28, output: 256 *14 * 14
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2) # conv layer, 3x3 kernels, input: 256 * 14 * 14, output: 512 * 7 * 7
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=7) # conv layer, 4x4 kernels, input: 512 * 7 * 7, output:  * 1 * 1
        
        self.conv_mean = nn.Conv2d(1024, z_dim, kernel_size=1)
        self.conv_var = nn.Conv2d(1024, z_dim, kernel_size=1)
        
        self.t_conv7 = nn.ConvTranspose2d(z_dim, 1024, kernel_size=1, stride=1)
        self.t_conv6 = nn.ConvTranspose2d(1024, 512, kernel_size=7, stride=1)
        self.t_conv5 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(64, 16, kernel_size=5, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2)
        
    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        mean = self.conv_mean(x)
        logvar = F.softplus(self.conv_var(x))
        return mean, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoder(self, z):
        x = F.relu(self.t_conv7(z))
        x = F.relu(self.t_conv6(x))
        x = F.relu(self.t_conv5(x))[:, :, :-1, :-1]
        x = F.relu(self.t_conv4(x))[:, :, :-1, :-1]
        x = F.relu(self.t_conv3(x))[:, :, 1:-2, 1:-2]
        x = F.relu(self.t_conv2(x))[:, :, 1:-2, 1:-2]
        x = torch.sigmoid(self.t_conv1(x))[:, :, 1:-2, 1:-2]
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# ### mnist

# In[ ]:


class ConvAE_mnist2(nn.Module):
    def  __init__(self, embedding_dimension):
        super().__init__()

        # encoder
        self.conv1 = nn.Sequential(nn.ZeroPad2d((1,2,1,2)),
                                                    nn.Conv2d(1, 32, kernel_size=5, stride=2),
                                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1,2,1,2)),
                                                    nn.Conv2d(32, 64, kernel_size=5, stride=2),
                                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
                                                    nn.ReLU())
        self.fc1 = nn.Conv2d(128, 10, kernel_size=3)

        # decoder
        self.fc2 = nn.Sequential(nn.ConvTranspose2d(10, 128, kernel_size=3),
                                               nn.ReLU())
        self.conv3d = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
                                                      nn.ReLU())
        self.conv2d = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
                                                      nn.ReLU())
        self.conv1d = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2)

    def forward(self, x):
        encoded = self.fc1(self.conv3(self.conv2(self.conv1(x))))

        decoded = self.fc2(encoded)
        decoded = self.conv3d(decoded)
        decoded = self.conv2d(decoded)[:,:,1:-2,1:-2]
        decoded = self.conv1d(decoded)[:,:,1:-2,1:-2]
        decoded = nn.Sigmoid()(decoded)

        return encoded, decoded


# In[1]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'vae_models.ipynb'])

