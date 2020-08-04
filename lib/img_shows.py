#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.manifold import TSNE, MDS, SpectralEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


def imdisp(poses, usgb10, ussp500, start_and_end, figsize=(15.0, 15.0)):
    plt.clf()
    fig = plt.figure(figsize=figsize)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    for i, pos in enumerate(poses):
        ax = fig.add_subplot(np.ceil(np.sqrt(len(poses))).astype("int64"), np.ceil(np.sqrt(len(poses))).astype("int64"), i + 1)
        # ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        # ax.tick_params(bottom=False, left=False, right=False, top=False)
        x = usgb10[start_and_end[pos][0]:start_and_end[pos][1]]
        y = ussp500[start_and_end[pos][0]:start_and_end[pos][1]]
        ax.plot(-x, y)
    fig.tight_layout()
    plt.show()


# In[ ]:


def imdisp_abs(poses, usgb10, ussp500, start_and_end, figsize=(15.0, 15.0)):
    plt.clf()
    fig = plt.figure(figsize=figsize)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    x_min, x_max = - usgb10.max(), - usgb10.min()
    y_min, y_max = ussp500.min(), ussp500.max()
    for i, pos in enumerate(poses):
        ax = fig.add_subplot(np.ceil(np.sqrt(len(poses))).astype("int64"), np.ceil(np.sqrt(len(poses))).astype("int64"), i + 1)
        # ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        # ax.tick_params(bottom=False, left=False, right=False, top=False)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        x = usgb10[start_and_end[pos][0]:start_and_end[pos][1]]
        y = ussp500[start_and_end[pos][0]:start_and_end[pos][1]]
        ax.plot(-x, y)
    fig.tight_layout()
    plt.show()


# In[ ]:


def imdisp_same_scale(poses, usgb10, ussp500, start_and_end, max_w, max_h, figsize=(15.0, 15.0)):
    plt.clf()
    fig = plt.figure(figsize=figsize)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    for i, pos in enumerate(poses):
        ax = fig.add_subplot(np.ceil(np.sqrt(len(poses))).astype("int64"), np.ceil(np.sqrt(len(poses))).astype("int64"), i + 1)
        # ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        # ax.tick_params(bottom=False, left=False, right=False, top=False)
        x = usgb10[start_and_end[pos][0]:start_and_end[pos][1]]
        y = ussp500[start_and_end[pos][0]:start_and_end[pos][1]]
        x_min = - x.max()
        y_min = y.min()
        ax.set_xlim([x_min, x_min + max_w])
        ax.set_ylim([y_min, y_min + max_h])
        ax.plot(-x, y)
    fig.tight_layout()
    plt.show()


# In[ ]:


def imdisp_gen_imgs(poses, usgb10, ussp500, dates, gen_imgs, max_w, max_h, lw=1.0, figsize=(15.0, 15.0)):
    plt.clf()
    fig = plt.figure(figsize=figsize)
    for i, pos in enumerate(poses):
        ax = fig.add_subplot(np.ceil(np.sqrt(len(poses))).astype("int64"), np.ceil(np.sqrt(len(poses))).astype("int64"), i + 1)
        img = gen_imgs(dates[pos][0], dates[pos][1], usgb10, ussp500, max_w, max_h, lw=lw)
        ax.imshow(img, cmap = 'gray', vmin = 0, vmax = 1, interpolation = 'none')
    fig.tight_layout()
    plt.show()


# In[ ]:


def imshow(dataset, dates, poses, model = None, device = "cpu", figsize=(15.0, 15.0), imgsize = 224):
    data_type = 1 if len(dataset[0].shape) == 1 else 2
    plt.clf()
    fig = plt.figure(figsize=figsize)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    for i, pos in enumerate(poses):
        ax = fig.add_subplot(np.ceil(np.sqrt(len(poses))).astype("int64"), np.ceil(np.sqrt(len(poses))).astype("int64"), i + 1)
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax.tick_params(bottom=False, left=False, right=False, top=False)
        ax.set_title(dates[pos])
        if model is None:
            if data_type == 1:
                ax.imshow(dataset[pos].to("cpu").reshape(imgsize, imgsize), cmap = 'gray', vmin = 0, vmax = 1, interpolation = 'none')
            else:
                ax.imshow(dataset[pos][0].to("cpu"), cmap = 'gray', vmin = 0, vmax = 1, interpolation = 'none')
        else:
            with torch.no_grad():
                output_img, _, _ = model(dataset[pos].unsqueeze(0))
                if data_type == 1:
                    ax.imshow(output_img.to("cpu").reshape(imgsize, imgsize), cmap = 'gray', vmin = 0, vmax = 1, interpolation = 'none')
                else:
                    ax.imshow(output_img.to("cpu")[0,0], cmap = 'gray', vmin = 0, vmax = 1, interpolation = 'none')
    fig.tight_layout()
    plt.show()


# In[ ]:


def imshow_cnn(dataset, dates, poses, device = "cpu", figsize=(15.0, 15.0), imgsize = 224):
    plt.clf()
    fig = plt.figure(figsize=figsize)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    for i, pos in enumerate(poses):
        ax = fig.add_subplot(np.ceil(np.sqrt(len(poses))).astype("int64"), np.ceil(np.sqrt(len(poses))).astype("int64"), i + 1)
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax.tick_params(bottom=False, left=False, right=False, top=False)
        ax.set_title(dates[pos])
        img, _ = dataset[pos]
        ax.imshow(img[0].to("cpu"), cmap = 'gray', vmin = 0, vmax = 1, interpolation = 'none')
    fig.tight_layout()
    plt.show()


# In[ ]:


def imshow_dense(dataset, poses, model = None, device = "cpu"):
    plt.clf()
    fig = plt.figure(figsize=(15.0, 75.0))
    for i, pos in enumerate(poses):
        ax = fig.add_subplot(1, len(poses), i + 1)
        if model is None:
            ax.imshow(dataset[pos].to("cpu").reshape(128,128), cmap = 'gray', vmin = 0, vmax = 1, interpolation = 'none')
        else:
            with torch.no_grad():
                output_img, _, _ = model(dataset[pos].unsqueeze(0).to(device))
                ax.imshow(output_img.to("cpu")[0].reshape(128,128), cmap = 'gray', vmin = 0, vmax = 1, interpolation = 'none')
    fig.tight_layout()
    plt.show()


# In[ ]:


def imshow_conv(dataset, poses, model = None, device = "cpu"):
    plt.clf()
    fig = plt.figure(figsize=(15.0, 75.0))
    for i, pos in enumerate(poses):
        ax = fig.add_subplot(1, len(poses), i + 1)
        if model is None:
            ax.imshow(dataset[pos][0].to("cpu"), cmap = 'gray', vmin = 0, vmax = 1, interpolation = 'none')
        else:
            with torch.no_grad():
                output_imgs, _, _ = model(dataset[pos].unsqueeze(0))
                ax.imshow(output_imgs.to("cpu")[0,0], cmap = 'gray', vmin = 0, vmax = 1, interpolation = 'none')
    fig.tight_layout()
    plt.show()


# In[ ]:


def visualize_zs(zs, labels, perplexity=30, learning_rate=200, n_iter=1000):
    plt.clf()
    plt.figure(figsize=(7,6))
    points = TSNE(n_components=2, random_state=0, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter).fit_transform(zs)
    plt.scatter(points.transpose()[0], points.transpose()[1], s=20, c=labels)
    plt.colorbar()
    plt.show()


# In[ ]:


def visualize_zsp(zs, perplexities, labels=None, learning_rate=200, n_iter=2000):
    plt.clf()
    fig = plt.figure(figsize=(len(perplexities) * 4, 4))
    for i, perplexity in enumerate(perplexities):
        points = TSNE(n_components=2, random_state=0, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter).fit_transform(zs)
        ax = fig.add_subplot(1, len(perplexities), i + 1)
        ax.scatter(points.transpose()[0], points.transpose()[1], s=5) if labels is None else ax.scatter(points.transpose()[0], points.transpose()[1], s=5, c=labels)
    fig.show()


# In[1]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'img_shows.ipynb'])

