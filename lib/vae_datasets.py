#!/usr/bin/env python
# coding: utf-8

# # Autoencoderのデータセットを定義

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import os
import pickle
from PIL import Image
from pprint import pprint
import random
import sys
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import datasets, transforms


# In[2]:


class autoencoder_dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, from_raw = False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_dir = img_dir
        self.data = self.make_monochrome_data() if from_raw else self.load_pkls()
        self.data_size = len(self.data)
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32).to(self.device)
    
    def to_monochrome(self, img):
        img_size = img.shape[0]
        new_img = np.zeros((img_size, img_size))
        for i in range(img_size):
            for j in range(img_size):
                new_img[i, j] = 0.0 if np.all(img[i, j] == 255) else 1.0
        return new_img
    
    def make_monochrome_data(self):
        data = []
        imgs1, imgs2, imgs3 = self.load_raw_pkls()
        
        print("imgs1 starts: ", len(imgs1))
        for i in range(len(imgs1)):
            print(i)
            data.append(self.to_monochrome(imgs1[i]["fig"]))
        print("pkl1 finished")
        
        print("imgs2 starts: ", len(imgs2))
        for i in range(len(imgs2)):
            print(i)
            data.append(self.to_monochrome(imgs2[i]["fig"]))
        print("pkl2 finished")
        
        print("imgs3 starts: ", len(imgs3))
        for i in range(len(imgs3)):
            print(i)
            data.append(self.to_monochrome(imgs3[i]["fig"]))
        print("pkl3 finished")
        
        with open("imgs.pkl", "wb") as pkl:
            pickle.dump(data, pkl)
        return data
    
    def load_pkls(self):
        with open("imgs.pkl", "rb") as pkl:
            data = pickle.load(pkl)
        return data
    
    def load_raw_pkls(self):
        with open(self.img_dir + "fig-1.pkl", "rb") as pkl1, open(self.img_dir + "fig-2.pkl", "rb") as pkl2, open(self.img_dir + "fig-3.pkl", "rb") as pkl3:
            imgs1 = pickle.load(pkl1)
            imgs2 = pickle.load(pkl2)
            imgs3 = pickle.load(pkl3)
        return imgs1, imgs2, imgs3


# In[10]:


class dataset2d(torch.utils.data.Dataset):
    def __init__(self, pkl_files):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pkl_files = pkl_files
        self.data = self.load_pkls()
        self.data_size = len(self.data)
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)[np.newaxis, :].to(self.device)

    def load_pkls(self):
        data = []
        for pkl_file in self.pkl_files:
            with open(pkl_file, "rb") as pkl:
                data += pickle.load(pkl)
        return data


# In[ ]:


class dataset1d(torch.utils.data.Dataset):
    def __init__(self, pkl_files):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pkl_files = pkl_files
        self.data = self.load_pkls()
        self.data_size = len(self.data)
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32).flatten().to(self.device)

    def load_pkls(self):
        data = []
        for pkl_file in self.pkl_files:
            with open(pkl_file, "rb") as pkl:
                data += pickle.load(pkl)
        return data


# In[ ]:


class dataset2d_alta(torch.utils.data.Dataset):
    def __init__(self, pkl_files):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pkl_files = pkl_files
        self.data = self.load_pkls()
        self.data = self.data[642:]
        self.data_size = len(self.data)
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)[np.newaxis, :].to(self.device)

    def load_pkls(self):
        data = []
        for pkl_file in self.pkl_files:
            with open(pkl_file, "rb") as pkl:
                data += pickle.load(pkl)
        return data


# In[3]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'vae_datasets.ipynb'])

