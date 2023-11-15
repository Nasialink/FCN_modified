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
#import cv2
from functions import *
tensorflow.compat.v1.disable_eager_execution() #my addition

#count = 284
patients=285
#patients = 20
#patients=10
count=patients
#count = patients - 1 
data_path = '/home/nasia/Documents/BRATS/data/MICCAI_BraTS_2018_Data_Training'
saving_path='/home/nasia/Documents/BRATS/data'

"""data_path='/content/gdrive/MyDrive/Diplomatiki_new/brats/MICCAI_BraTS_2018_Data_Training'
saving_path='/content/gdrive/MyDrive/Diplomatiki_new/brats'"""

# Get a list of files for all modalities individually
t1 = glob.glob(data_path+'/*GG/*/*t1.nii.gz')
t2 = glob.glob(data_path+'/*GG/*/*t2.nii.gz')
flair = glob.glob(data_path+'/*GG/*/*flair.nii.gz')
t1ce = glob.glob(data_path+'/*GG/*/*t1ce.nii.gz')
seg = glob.glob(data_path+'/*GG/*/*seg.nii.gz')  # Ground Truth

print(len(seg))

pat = re.compile('.*_(\w*)\.nii\.gz')

data_paths = [{
    pat.findall(item)[0]:item
    for item in items
}
for items in list(zip(t1, t2, t1ce, flair, seg))]

print(data_paths[5:16])

"""## Load the data in a Numpy array
#Creating an empty Numpy array beforehand and then filling up the data helps you gauge beforehand if the data fits in your memory.

#Loading only the first 4 images here, to save time._
"""
input_shape = (4, 80, 96, 64)
#input_shape= (4, 160, 192, 128)
output_channels = 3
data = np.empty((len(data_paths[:count]),) + input_shape, dtype=np.float32)
labels = np.empty((len(data_paths[:count]), output_channels) + input_shape[1:], dtype=np.uint8)


print(data.shape, labels.shape)
# Parameters for the progress bar
total = len(data_paths[:count])
step = 25 / total

for i, imgs in enumerate(data_paths[:count]):
    try:
        data[i] = np.array([preprocess(read_img(imgs[m]), input_shape[1:]) for m in ['t1', 't2', 't1ce', 'flair']], dtype=np.float32)
        labels[i] = preprocess_label(read_img(imgs['seg']), input_shape[1:])[None, ...]
        
        # Print the progress bar
        print('\r' + f'Progress: '
            f"[{'=' * int((i+1) * step) + ' ' * (24 - int((i+1) * step))}]"
            f"({math.ceil((i+1) * 100 / (total))} %)",
            end='')
    except Exception as e:
        print(f'Something went wrong with {imgs["t1"]}, skipping...\n Exception:\n{str(e)}')
        continue
    
print(np.unique(labels))    
all_data= nib.Nifti1Image(data, affine=np.eye(4)) #converting np.arrays into nii.gz files
all_lab= nib.Nifti1Image(labels, affine=np.eye(4))

nib.save(all_data, os.path.join(saving_path, 'data.nii.gz'))
nib.save(all_lab, os.path.join(saving_path,'labels.nii.gz'))
