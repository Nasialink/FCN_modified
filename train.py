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
'''
data_path = '/home/nasia/Documents/BRATS/data/data.nii.gz'
labels_path = '/home/nasia/Documents/BRATS/data/labels.nii.gz'
root_path='/home/nasia/Documents/BRATS/data/training_'+ datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
#path_checkpoint = root_path +'/cp.ckpt'
save_pred_path = root_path+'/predictions'
save_model_path = root_path+'/saved_model'
test_file = root_path+'/testset'


data_path = '/media/nasia/ssd2tb/nasia/data/data_one.nii.gz'
labels_path = '/media/nasia/ssd2tb/nasia/data/labels_one.nii.gz'
root_path='/media/nasia/ssd2tb/nasia/training_'+ datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
#path_checkpoint = root_path +'/cp.ckpt'
save_pred_path = root_path+'/predictions'
save_model_path = root_path+'/saved_model'
test_file = root_path+'/testset'
'''
data_path = '/home/azach/FCN_modified_new/data/data_cropped.nii.gz'
labels_path = '/home/azach/FCN_modified_new/data/labels_cropped.nii.gz'
root_path='/home/azach/FCN_modified_new/data/training_'+ datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
#path_checkpoint = root_path +'/cp.ckpt'
save_pred_path = root_path+'/predictions'
save_model_path = root_path+'/saved_model'
test_file = root_path+'/testset'

dirs=['training_new','predictions','saved_model','testset']
os.mkdir(root_path)
for directory in dirs:
    os.mkdir(os.path.join(root_path, directory))

print(tensorflow.test.is_gpu_available())
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


"""data_path = '/content/gdrive/MyDrive/Diplomatiki_new/brats/data_one.nii.gz'
labels_path = '/content/gdrive/MyDrive/Diplomatiki_new/brats/labels_one.nii.gz'
path_checkpoint = '/content/gdrive/MyDrive/Diplomatiki_new/brats/training/'
save_pred_path = '/content/gdrive/MyDrive/Diplomatiki_new/brats/predictions'
save_model_path = '/content/gdrive/MyDrive/Diplomatiki_new/brats/saved_model'
test_idx_file = '/content/gdrive/MyDrive/Diplomatiki_new/brats/testset'+datetime.datetime.now().strftime("%m%d%Y_%H%M%S")+'.json'"""
#count = 20
#count = 10σ
batch_size = 1
epochs = 100
period = 10
input_shape = (4, 160, 192, 128)
#input_shape = (4, 80, 96, 64)
output_channels = 3
json_file_path = root_path+'/history.json'

lrn=0.00001

def scheduler(epoch):
    lr_new= lrn* ((1-(epoch/epochs))**0.9)
    return lr_new

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        return json.JSONEncoder.default(self, obj)
    
np_data = nib.load(data_path).get_fdata()
np_labels = nib.load(labels_path).get_fdata()
print(np_data.shape)
print(np.max(np_data), np.min(np_data))
# data = np.empty(data.shape[0] + input_shape, dtype=np.float32)
# labels = np.empty((data.shape[0]), output_channels) + input_shape[1:], dtype=np.uint8)

# for i in data.shape[0]:
#     data = sitk.GetArrayFromImage(data)
# labels = sitk.GetArrayFromImage(labels)

# Split into training and test set 
data_train, data_test, labels_train, labels_test = train_test_split(np_data, np_labels, test_size=0.25, random_state=None)
print(data_train.shape, data_test.shape, labels_train.shape, labels_test.shape)
print(np.max(labels_train), np.max(labels_test), np.min(labels_train),np.min(labels_test))

test_d= nib.Nifti1Image(data_test, affine=np.eye(4)) #converting np.arrays into nii.gz files
test_l= nib.Nifti1Image(labels_test, affine=np.eye(4)) 

nib.save(test_d, os.path.join(test_file, 'data_test.nii.gz'))
nib.save(test_l, os.path.join(test_file, 'labels_test.nii.gz'))

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

#directory_checkpoint = os.path.dirname(path_checkpoint)

#cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,save_weights_only=True,verbose=1)
cp_callback = keras.callbacks.ModelCheckpoint(filepath=root_path + '/cp_{epoch:002d}'+'.cpkt',save_weights_only=True,verbose=1,period=period)#save_freq=430) =train data size/batch size * frequency (π.χ.ανά 10 εποχές)

#callbacks= [keras.callbacks.ModelCheckpoint(filepath=root_path + '/cp_{epoch:002d}'+'.cpkt',save_weights_only=True,verbose=1, period=period),keras.callbacks.LearningRateScheduler(scheduler)]

"""Train the model"""

history_dict = {}
#history= model.fit(data_train, [labels_train, data_train], validation_split=0.15, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
history= model.fit(data_train, [labels_train, data_train], validation_split=0.15, batch_size=batch_size, epochs=epochs, callbacks=[cp_callback])
history_dict=history.history
for key,value in history_dict.items():
    print(key,type(history_dict[key]))
    for i in range(len(value)):
        value[i] = np.float64(value[i])
"""for key, value in history_dict.items():
    history_dict[key] = list(value)"""

#history_json = json.dumps(history_dict, cls=NumpyEncoder)

# Save the JSON string to a file
"""with open(json_file_path, 'w') as json_file:
    json_file.write(history_json)"""

with open(json_file_path, 'w') as json_file:
    json.dump(history_dict, json_file)
    #history_json = json.dumps(history_dict, cls=NumpyEncoder)

#model.load_weights(path_checkpoint)
# os.makedirs('saved_model', exist_ok=True)

model.save(os.path.join(save_model_path,'mymodel.keras'))
