#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import datetime
import io
from IPython.display import display, Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import pickle
import scipy
from scipy import interpolate
from tqdm import tqdm


# In[ ]:


def make_date_set(year_interval = 0, month_interval = 0, day_interval = 0):
    years = [1962 + i  for i  in range(0, 58) ] # 1962~2020まで
    months = [ i+1  for i  in range(0, 12)  ] # 1~12月
    days = [ i+1  for i  in range(0, 31) ] # 1ヶ月が31日あると仮定
    start_and_end = list()
    for year in years:
        for month in months:
            for day in days:
                try:
                    start = datetime.date(year, month, day)
                    if month != 12:
                        end  = datetime.date(year+year_interval, month+month_interval, day+day_interval)
                    else: # month==12の場合は、次の年のmonth_interval月を終点とする        
                        end  = datetime.date(year+year_interval+1, month_interval, day+day_interval)
                        start_and_end.append( ( str(start), str(end) ) )
                except Exception as e: # 条件を満たさないものは無視
                    pass
    return start_and_end


# In[ ]:


def gen_dates():
    years = [1962 + i  for i  in range(58) ]
    months = [ i+2  for i  in range(12)  ]
    days = [ i+1  for i  in range(31) ]

    start_and_end = []
    for year in years:
        for month in months:
            for day in days:
                try:
                    start = datetime.date(year, month, day)
                    if month != 12:
                        end  = datetime.date(year, month+1, day)
                    else:
                        end  = datetime.date(year+1, 1, day)
                    start_and_end.append( ( str(start), str(end) ) )
                except Exception as e:
                    pass
    return start_and_end


# In[ ]:


def to_monochrome(img):
    img_size = img.shape[0]
    new_img = np.zeros((img_size, img_size))
    for i in range(img_size):
        for j in range(img_size):
            new_img[i, j] = 0.0 if np.all(img[i, j] == 255) else 1.0
    return new_img


# In[ ]:


def gen_imgs_scaled(start, end, usgb10, ussp500, size = 64, lw = 1.0):
    # fig setting
    fig = plt.figure(dpi=1, figsize=(size, size))
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    
    # read data
    x = usgb10[start:end]
    y = ussp500[start:end]
    
    # plot
    # plt.plot(-x, y, marker=".")
    plt.plot(-x, y, lw=lw)
    # プロット画像を直接メモリで渡す                                                   
    buf = io.BytesIO() # bufferを用意
    plt.savefig(buf, format='png') # bufferに保持
    enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # bufferからの読み出し
    dst = cv2.imdecode(enc, 1) # デコード
    dst = dst[:,:,::-1] # BGR->RGB
    plt.close()
    img = to_monochrome(dst)
    return img


# In[ ]:


def spline(x,y,point,deg):
    tck,u = interpolate.splprep([x,y],k=deg,s=0) 
    u = np.linspace(0,1,num=point,endpoint=True) 
    spline = interpolate.splev(u,tck)
    return spline[0],spline[1]


# In[ ]:


def gen_imgs_scaled_spline(start, end, usgb10, ussp500, size = 64, lw = 1.0):
    # fig setting
    fig = plt.figure(dpi=1, figsize=(size, size))
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    
    # read data
    x = usgb10[start:end]
    y = ussp500[start:end]
    
    # plot
    # plt.plot(-x, y, marker=".")
    a, b = spline(-x, y, 100, 2)
    plt.plot(a, b, lw=lw)
    # プロット画像を直接メモリで渡す                                                   
    buf = io.BytesIO() # bufferを用意
    plt.savefig(buf, format='png') # bufferに保持
    enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # bufferからの読み出し
    dst = cv2.imdecode(enc, 1) # デコード
    dst = dst[:,:,::-1] # BGR->RGB
    plt.close()
    img = to_monochrome(dst)
    return img


# In[ ]:


def find_hw(usgb10, ussp500, dates):
    max_h, max_w = 0, 0
    for i in range(len(dates)):
        x = usgb10[dates[i][0]:dates[i][1]]
        y = ussp500[dates[i][0]:dates[i][1]]
        max_w = x.max() - x.min() if (x.max() - x.min()) > max_w else max_w
        max_h = y.max() - y.min() if (y.max() - y.min()) > max_h else max_h
    return max_w, max_h


# In[ ]:


def gen_imgs(start, end, usgb10, ussp500, max_w, max_h, size = 224, lw = 1.0):
    # fig setting
    fig = plt.figure(dpi=1, figsize=(size, size))
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    
    # read data
    x = usgb10[start:end]
    y = ussp500[start:end]
    
    x_min = - x.max()
    y_min = y.min()
    plt.xlim([x_min, x_min + max_w])
    plt.ylim([y_min, y_min + max_h])
    # plot
    # plt.plot(-x, y, marker=".")
    plt.plot(-x, y, lw=lw)
    # プロット画像を直接メモリで渡す                                                   
    buf = io.BytesIO() # bufferを用意
    plt.savefig(buf, format='png') # bufferに保持
    enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # bufferからの読み出し
    dst = cv2.imdecode(enc, 1) # デコード
    dst = dst[:,:,::-1] # BGR->RGB
    plt.close()
    img = to_monochrome(dst)
    return img


# In[3]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'gen_functions.ipynb'])

