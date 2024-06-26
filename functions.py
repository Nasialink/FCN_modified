# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1asf3jaPSOrlrPWxlNnJl1Tyr7onXVqMX
"""

import numpy as np
import SimpleITK as sitk 
from scipy.ndimage import zoom
import cv2
import nibabel as nib

def read_img(img_path):
    """
    Reads a .nii.gz image and returns as a numpy array.
    """
    #return sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    return nib.load(img_path).get_fdata()

def resize(img, shape, mode='constant', orig_shape=(155, 240, 240)):
    """
    Wrapper for scipy.ndimage.zoom suited for MRI images.
    """
    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        shape[0]/orig_shape[0],
        shape[1]/orig_shape[1],
        shape[2]/orig_shape[2]
    )

    # Resize to the given shape
    return zoom(img, factors, mode=mode)


def preprocess(img, out_shape=None):
    """
    Preprocess the image.
    Just an example, you can add more preprocessing steps if you wish to.
    """
    if out_shape is not None:
        #img = resize(img, out_shape, mode='constant')
        img= random_crop(img, out_shape)
    return img

    # Normalize the image
'''
    mean = img.mean()
    std = img.std()
    return (img - mean) / std
'''

def preprocess_label(img, out_shape=None, mode='nearest'):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)"""

    kernel = np.ones((3, 3))   ##preprocessing suggestions
    #img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3) #or
    img_dilated = cv2.dilate(img, kernel, iterations=1)
    #img=img_closed
    img=img_dilated

    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = img == 2  # Peritumoral Edema (ED)
    et = img == 4  # GD-enhancing Tumor (ET)

    if out_shape is not None:
        """
        ncr = resize(ncr, out_shape, mode=mode)
        ed = resize(ed, out_shape, mode=mode)
        et = resize(et, out_shape, mode=mode)
"""     
        ncr = random_crop(ncr, out_shape)
        ed = random_crop(ed, out_shape)
        et = random_crop(et, out_shape)

    output=np.array([ncr, ed, et], dtype=np.uint8)
    print(output.shape)
    return output

def random_crop(img, output_size):
    """
    Randomly crop a NIfTI image to the specified output size.

    Parameters:
    - input_path: str, path to the input NIfTI file.
    - output_path: str, path to save the cropped NIfTI file.
    - output_size: tuple of ints, desired output size (depth, height, width).

    Returns:
    - cropped_img: numpy array, the cropped image data.
    """
    # Load the NIfTI image
    #nii_img = nib.load(input_path)
    #img_data = nii_img.get_fdata()
    
    # Get the dimensions of the input image
    height, width, depth = img.shape

    # Desired output dimensions
    output_height, output_width, output_depth = output_size

    # Ensure the output size is smaller than the input size
    assert output_depth <= depth and output_height <= height and output_width <= width, \
        "Output size must be smaller than the input size."
        # Calculate the center of the image
    center_d = depth // 2
    center_h = height // 2
    center_w = width // 2

    # Calculate the starting point for the crop
    start_d = max(0, center_d - output_depth // 2)
    start_h = max(0, center_h - output_height // 2)
    start_w = max(0, center_w - output_width // 2)
    cropped_img = img[start_h:start_h + output_height, start_w:start_w + output_width, start_d:start_d + output_depth]
    print(cropped_img.shape)

    return cropped_img
'''
    # Ensure the crop window does not exceed the image dimensions
    start_d = min(start_d, depth - output_depth)
    start_h = min(start_h, height - output_height)
    start_w = min(start_w, width - output_width)
'''
    # Perform the crop
    

    # Create a new NIfTI image from the cropped data
    #cropped_nii = nib.Nifti1Image(cropped_img, nii_img.affine, nii_img.header)
   
