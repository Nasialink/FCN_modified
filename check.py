import os

import math
import matplotlib.pyplot as plt
import numpy as np  # For data ma

import nibabel as nib
import random

data_path = '/home/superteam/superproject/data/data_cropped.nii.gz'
labels_path = '/home/superteam/superproject/data/labels_cropped.nii.gz'
# root_path='/home/azach/FCN_modified_new/data/training_'+ datetime.datetime.now().strftime("%m%d%Y_%H%M%S")






batch_size = 1
epochs = 100
period = 10
input_shape = (4, 160, 192, 128)
#input_shape = (4, 80, 96, 64)
output_channels = 3

    
np_data = nib.load(data_path).get_fdata()
lb_data = nib.load(labels_path).get_fdata()

print(np_data.shape)
print(np.min(np_data), np.max(np_data))
print(lb_data.shape)
print(np.min(lb_data), np.max(lb_data))

for i in range(25):
    p = random.randint(0, len(np_data))
    c = random.randint(0, 3)
    s = random.randint(30, 90)
    plt.figure()
    plt.imshow(np_data[p, c, :, :, s], cmap='gray')
    plt.savefig('check/'+ str(p) +'_'+ str(c) +'_'+ str(s) +'_im.png')
    plt.close()

    plt.figure()
    plt.imshow(lb_data[p, 0, :, :, s], cmap='gray')
    plt.savefig('check/'+ str(p) +'_'+ str(c) +'_'+ str(s) +'_lb.png')
    plt.close()

    plt.figure()
    plt.imshow(lb_data[p, 1, :, :, s], cmap='gray')
    plt.savefig('check/'+ str(p) +'_'+ str(c) +'_'+ str(s) +'_lb_1.png')
    plt.close()

    plt.figure()
    plt.imshow(lb_data[p, 2, :, :, s], cmap='gray')
    plt.savefig('check/'+ str(p) +'_'+ str(c) +'_'+ str(s) +'_lb_2.png')
    plt.close()
