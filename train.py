import os
import tensorflow
import keras
import math
import matplotlib.pyplot as plt
import zipfile  # For faster extraction
import SimpleITK as sitk  # For loading the dataset
import numpy as np  # For data manipulation
from model import build_model  # For creating the model
import glob  # For populating the list of files
from scipy.ndimage import zoom  # For resizing
import re  # For parsing the filenames (to know their modality)
from datetime import datetime
from PIL import Image
#import cv2
from functions import *
from sklearn.model_selection import train_test_split
import json 
import datetime
import nibabel as nib
tensorflow.compat.v1.disable_eager_execution() #my addition

data_path = '/home/nasia/Documents/BRATS/data/data.nii.gz'
labels_path = '/home/nasia/Documents/BRATS/data/labels.nii.gz'
path_checkpoint = '/home/nasia/Documents/BRATS/data/training/cp.ckpt'
save_pred_path = '/home/nasia/Documents/BRATS/data/predictions'
save_model_path = '/home/nasia/Documents/BRATS/data/saved_model'
test_idx_file = '/home/nasia/Documents/BRATS/data/testset'+datetime.datetime.now().strftime("%m%d%Y_%H%M%S")+'.json'

"""data_path = '/content/gdrive/MyDrive/Diplomatiki_new/brats/data.nii.gz'
labels_path = '/content/gdrive/MyDrive/Diplomatiki_new/brats/labels.nii.gz'
path_checkpoint = '/content/gdrive/MyDrive/Diplomatiki_new/brats/training/cp.ckpt'
save_pred_path = '/content/gdrive/MyDrive/Diplomatiki_new/brats/predictions'
save_model_path = '/content/gdrive/MyDrive/Diplomatiki_new/brats/saved_model'
test_idx_file = '/content/gdrive/MyDrive/Diplomatiki_new/brats/testset'+datetime.datetime.now().strftime("%m%d%Y_%H%M%S")+'.json'"""
#count = 20
#count = 10
batch_size = 4
epochs = 50

data = nib.load(data_path)
labels = nib.load(labels_path)

print(data.shape)
print(labels.shape)
input_shape = (4, 80, 96, 64)
output_channels = 3
np_data = data.get_fdata()
np_labels = labels.get_fdata()
# data = np.empty(data.shape[0] + input_shape, dtype=np.float32)
# labels = np.empty((data.shape[0]), output_channels) + input_shape[1:], dtype=np.uint8)
# np_data
# for i in data.shape[0]:
#     data = sitk.GetArrayFromImage(data)
# labels = sitk.GetArrayFromImage(labels)

# Split into training and test set 
data_train, data_test, labels_train, labels_test = train_test_split(np_data, np_labels, test_size=0.25, random_state=None)
print(data_train.shape, data_test.shape, labels_train.shape, labels_test.shape)
print(np.max(labels_train), np.max(labels_test), np.min(labels_train),np.min(labels_test))

"""all_indices = list(range(count))
train_ind, test_ind = train_test_split(all_indices, test_size=0.25)

data_train = np_data[train_ind,:,:,:,:]
labels_train= np_labels[train_ind,:,:,:,:]

np.save(os.path.join(save_pred_path, 'test_ind.npy'), test_ind)"""
# data = data[data_train]
# labels = labels[labels_train]
# testset = {'data_test': data_test, 'labels_test': labels_test}
# with open(test_idx_file, 'w') as f:
#     f.write(json.dump(testset), indent=4)

## Model

#build the model

model = build_model(input_shape=input_shape, output_channels=3)

#model.summary()

directory_checkpoint = os.path.dirname(path_checkpoint)

#cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,save_weights_only=True,verbose=1)
cp_callback = keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,save_weights_only=True,verbose=1)

"""Train the model"""

model.fit(data_train, [labels_train, data_train], batch_size=batch_size, epochs=epochs, callbacks=[cp_callback])

#model.load_weights(path_checkpoint)
os.makedirs('saved_model', exist_ok=True)
model.save(os.path.join(save_model_path,'mymodel.keras'))