#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import torch
from skimage import io, transform


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


# In[3]:


# torch.cuda.device_count()


# In[4]:


class cnn_dataset_to224(torch.utils.data.Dataset):
    def __init__(self, pkl_imgs, pkl_labels, classes, label_onehot=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = self.load_pkls(pkl_imgs)
        self.data_size = len(self.data)
        self.labels = self.load_pkl(pkl_labels)
        self.classes = classes
        self.label_onehot = label_onehot
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        img = self.resize_img(self.data[index])
        img = torch.tensor(img, dtype=torch.float32).expand(3, img.shape[0], img.shape[1]) # (-1, 3, -1, -1)
        return (img.to(self.device), torch.eye(self.classes)[self.labels[index]]) if self.label_onehot else (img.to(self.device), self.labels[index])

    def load_pkls(self, pkls):
        data = []
        for pkl_file in pkls:
            with open(pkl_file, "rb") as pkl:
                data += pickle.load(pkl)
        return data
    
    def load_pkl(self, pkl_file):
        with open(pkl_file, "rb") as pkl:
            return pickle.load(pkl)
        
    def resize_img(self, img):
        return transform.resize(img, (224, 224))


# In[ ]:


class cnn_dataset(torch.utils.data.Dataset):
    def __init__(self, pkl_imgs, pkl_labels, classes, label_onehot=False, interval = 1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = self.load_pkls(pkl_imgs)
        self.data = self.data[::interval]
        self.data_size = len(self.data)
        self.labels = self.load_pkl(pkl_labels)
        self.labels = self.labels[::interval]
        self.classes = classes
        self.label_onehot = label_onehot
        self.interval = interval
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        img = torch.tensor(self.data[index], dtype=torch.float32).expand(3, self.data[index].shape[0], self.data[index].shape[1]) # (-1, 3, -1, -1)
        return (img.to(self.device), torch.eye(self.classes)[self.labels[index]]) if self.label_onehot else (img.to(self.device), self.labels[index])

    def load_pkls(self, pkls):
        data = []
        for pkl_file in pkls:
            with open(pkl_file, "rb") as pkl:
                data += pickle.load(pkl)
        return data
    
    def load_pkl(self, pkl_file):
        with open(pkl_file, "rb") as pkl:
            return pickle.load(pkl)


# In[1]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'cnn_dataset.ipynb'])

