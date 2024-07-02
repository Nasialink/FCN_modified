# -*- coding: utf-8 -*-
"""Preprocess.ipynb
"""
import os
import re
import math
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk  # For loading the dataset

from functions import *

#count = 284
patients=285+50+34 #brats_2018 + brats_2019 + brats_2020
#patients=285
patients=40
count=patients

data_path = ['/home/azach/testdir/raw_data/MICCAI_BraTS_2018','/home/azach/testdir/raw_data/BRATS_2019', '/home/azach/testdir/raw_data/BRATS_2020']
saving_path='/home/azach/testdir/data'
t1_all,t2_all,t1ce_all,flair_all,seg_all=[],[],[],[],[]
input_shape= (4, 160, 160, 90) 
output_channels = 2
max_train, min_train=[],[]

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


pat = re.compile('.*_(\w*)\.nii\.gz')

data_paths = [{
    pat.findall(item)[0]:item
    for item in items
}
for items in list(zip(t1_all, t2_all, t1ce_all, flair_all, seg_all))]


data = np.empty((len(data_paths[:count]),) + input_shape, dtype=np.float32)
labels = np.empty((len(data_paths[:count]), output_channels) + input_shape[1:], dtype=np.uint8)

print(data.shape, labels.shape)
# Parameters for the progress bar
total = len(data_paths[:count])
step = 25 / total

for i, imgs in enumerate(data_paths[:count]):
    # print(i, imgs)
    # if i > 1:
    #     break
    # try:
    data[i] = np.array([preprocess(read_img(imgs[m])) for m in ['t1', 't2', 't1ce', 'flair']], dtype=np.float32)
    labels[i] = preprocess_label(read_img(imgs['seg']))
    
    # Print the progress bar
    print('\r' + f'Progress: '
        f"[{'=' * int((i+1) * step) + ' ' * (24 - int((i+1) * step))}]"
        f"({math.ceil((i+1) * 100 / (total))} %)",
        end='')
    # except Exception as e:
    #     print(f'Something went wrong with {imgs["t1"]}, skipping...\n Exception:\n{str(e)}')
    #     continue

    

print(max_train,min_train)
print(np.max(data[:,1,:,:,:]),np.max(data[:,2,:,:,:]),np.max(data[:,3,:,:,:]))

# 185, 4, 160, 192, 128
print(data.shape)
for w in range(data.shape[1]):
    min_c = np.min(data[:, w, :, :, :])
    max_c = np.max(data[:, w, :, :, :])

    data[:, w, :, :, :] = (data[:, w, :, :, :] - min_c) / (max_c - min_c)


print(np.unique(labels)) 
print("Min data: ", np.min(data), "Max data: ", np.max(data))   
all_data= nib.Nifti1Image(data, affine=np.eye(4)) #converting np.arrays into nii.gz files
all_lab= nib.Nifti1Image(labels, affine=np.eye(4))

nib.save(all_data, os.path.join(saving_path, 'data_cropped.nii.gz'))
nib.save(all_lab, os.path.join(saving_path,'labels_cropped.nii.gz'))
