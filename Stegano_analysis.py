# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:31:51 2020

@author: sharm
"""

import stegano
from stegano import lsb
#System
import cv2
import os,os.path
from PIL import Image
#Basic libraries
import pandas as pd
import numpy as np
# discrete cosine tranform
from numpy import pi
from numpy import r_
import scipy
from scipy import fftpack
import random
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
# sklearn
from sklearn.model_selection import KFold
from sklearn import metrics
#pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# Data augmentation for image preprocessing
from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, 
                            Compose, Resize, RandomBrightness, RandomContrast, HueSaturationValue, Blur, GaussNoise)

from albumentations.pytorch import ToTensorV2, ToTensor
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet34

import warnings
warnings.filterwarnings("ignore")

# EDA and reading the data
base_path = 'D:\Kaggle Project'

def read_images_path(dir_name = 'Cover',test = False):
    series_name = pd.Series(os.listdir(base_path + '/' + dir_name))
    if test:
        series_name = pd.Series(os.listdir(base_path + '/' + 'Test'))
        
    series_paths = pd.Series(base_path + '/' + dir_name + '/' + series_name)
    
    return series_paths

# Read in the data
cover_paths = read_images_path('Cover', False)
jmipod_paths = read_images_path('JMiPOD', False)
juniward_paths = read_images_path('JUNIWARD', False)
uerd_paths = read_images_path('UERD', False)
test_paths = read_images_path('Test', True)


def show15(title = "Default"):
    plt.figure(figsize=(16,9))
    plt.suptitle(title, fontsize = 16)
    
    for k, path in enumerate(cover_paths[:15]):
        cover = mpimg.imread(path)
        
        plt.subplot(3, 5, k+1)
        plt.imshow(cover)
        plt.axis('off')
        
image_sample = mpimg.imread(cover_paths[0])

print('Image sample shape:', image_sample.shape)
print('Image sample size:', image_sample.size)
print('Image sample data type:', image_sample.dtype)


def show_images_alg(n=3,title = "Default"):
    f,ax = plt.subplots(n,4,figsize=(16,7))
    plt.suptitle(title,fontsize=16)
    
    for k,path in enumerate(cover_paths[:15]):
        cover = mpimg.imread(path)
        
        plt.subplot(3,5,k+1)
        plt.imshow(cover)
        plt.axis('off')
        
show15(title= "15 original images")

# algorithms
# JMiPOD, juniward and uerd

def show_images_alg(n=3,title = "Default"):
    
    f,ax = plt.subplots(n,4,figsize=(16,7))
    plt.suptitle(title,fontsize=16)
    
    for index in range(n):
        cover = mpimg.imread(cover_paths[index])
        ipod = mpimg.imread(jmipod_paths[index])
        juni = mpimg.imread(juniward_paths[index])
        uerd = mpimg.imread(uerd_paths[index])
        
        
        # plot
        ax[index, 0].imshow(cover)
        ax[index,1].imshow(ipod)
        ax[index,2].imshow(juni)
        ax[index,3].imshow(uerd)
        
show_images_alg(n=3,title="Algorithm Difference")
