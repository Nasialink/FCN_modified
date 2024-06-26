# -*- coding: utf-8 -*-
"""Preprocess.ipynb
"""
import os
import tensorflow
import keras
import math
import matplotlib.pyplot as plt
import SimpleITK as sitk  # For loading the dataset
import numpy as np  # For data manipulation
import nibabel as nib
from model import build_model  # For creating the model
import glob  # For populating the list of files
from scipy.ndimage import zoom  # For resizing
import re  # For parsing the filenames (to know their modality)
from datetime import datetime
from PIL import Image
import cv2
from functions import *
tensorflow.compat.v1.disable_eager_execution() #my addition

#count = 284
patients=285+50+34 #brats_2018 + brats_2019 + brats_2020
#patients=285
#patients=26
count=patients

#data_path = ['/home/nasia/Documents/BRATS/data/MICCAI_BraTS_2017_Training_Data','/home/nasia/Documents/BRATS/data/MICCAI_BraTS_2018_Data_Training', '/home/nasia/Documents/BRATS/data/MICCAI_BraTS_2019_Training_Dataset']
#data_path = ['/media/nasia/ssd2tb/nasia/MICCAI_BraTS_2018', '/media/nasia/ssd2tb/nasia/BRATS_2019','/media/nasia/ssd2tb/nasia/BRATS_2020']
data_path = ['/home/superteam/superproject/raw_data/MICCAI_BraTS_2018','/home/superteam/superproject/raw_data/BRATS_2019', '/home/superteam/superproject/raw_data/BRATS_2020']
saving_path='/home/superteam/superproject/data'
t1_all,t2_all,t1ce_all,flair_all,seg_all=[],[],[],[],[]
input_shape= (4, 160, 192, 128) 
output_channels = 3
max_train, min_train=[],[]
"""data_path='/content/gdrive/MyDrive/Diplomatiki_new/brats/MICCAI_BraTS_2018_Data_Training'
saving_path='/content/gdrive/MyDrive/Diplomatiki_new/brats'"""

# Get a list of files for all modalities individually
for i in data_path:
    t1 = glob.glob(i+'/*GG/*/*t1.nii.gz')
    t2 = glob.glob(i+'/*GG/*/*t2.nii.gz')
    flair = glob.glob(i+'/*GG/*/*flair.nii.gz')
    t1ce = glob.glob(i+'/*GG/*/*t1ce.nii.gz')
    seg = glob.glob(i+'/*GG/*/*seg.nii.gz')  # Ground Truth
    t1_all=t1_all+t1
    t2_all=t2_all+t2
    flair_all=flair_all+flair
    t1ce_all=t1ce_all+t1ce
    seg_all=seg_all+seg
print(len(seg_all))

"""    
t1 = glob.glob(data_path+'/*GG/*/*t1.nii.gz')
t2 = glob.glob(data_path+'/*GG/*/*t2.nii.gz')
flair = glob.glob(data_path+'/*GG/*/*flair.nii.gz')
t1ce = glob.glob(data_path+'/*GG/*/*t1ce.nii.gz')
seg = glob.glob(data_path+'/*GG/*/*seg.nii.gz')  # Ground Truth
"""

pat = re.compile('.*_(\w*)\.nii\.gz')

data_paths = [{
    pat.findall(item)[0]:item
    for item in items
}
for items in list(zip(t1_all, t2_all, t1ce_all, flair_all, seg_all))]

#print(data_paths[5:16])

#input_shape = (4, 80, 96, 64)

data = np.empty((len(data_paths[:count]),) + input_shape, dtype=np.float32)
labels = np.empty((len(data_paths[:count]), output_channels) + input_shape[1:], dtype=np.uint8)

print(data.shape, labels.shape)
# Parameters for the progress bar
total = len(data_paths[:count])
step = 25 / total

for i, imgs in enumerate(data_paths[:count]):
    try:
        data[i] = np.array([preprocess(read_img(imgs[m]), input_shape[1:]) for m in ['t1', 't2', 't1ce', 'flair']], dtype=np.float32)
        labels[i] = preprocess_label(read_img(imgs['seg']), input_shape[1:])#[None, ...]
        
        # Print the progress bar
        print('\r' + f'Progress: '
            f"[{'=' * int((i+1) * step) + ' ' * (24 - int((i+1) * step))}]"
            f"({math.ceil((i+1) * 100 / (total))} %)",
            end='')
    except Exception as e:
        print(f'Something went wrong with {imgs["t1"]}, skipping...\n Exception:\n{str(e)}')
        continue
print(data[1,1,:,:,:])
print(data[1,1,:,:,:].shape)

for i in range(data.shape[1]):
    max_train.append(np.max(data[:,i,:,:,:]))
    min_train.append(np.min(data[:,i,:,:,:]))
    

print(max_train,min_train)
print(np.max(data[:,1,:,:,:]),np.max(data[:,2,:,:,:]),np.max(data[:,3,:,:,:]))

min_train = np.array(min_train).reshape(4, 1, 1, 1)
max_train = np.array(max_train).reshape(4, 1, 1, 1)

for img in data:
    img = (img - min_train) / (max_train - min_train)

print(np.unique(labels))    
all_data= nib.Nifti1Image(data, affine=np.eye(4)) #converting np.arrays into nii.gz files
all_lab= nib.Nifti1Image(labels, affine=np.eye(4))

nib.save(all_data, os.path.join(saving_path, 'data_cropped.nii.gz'))
nib.save(all_lab, os.path.join(saving_path,'labels_cropped.nii.gz'))
