
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def read_img(img_path):
    return nib.load(img_path).get_fdata()


def preprocess(img):
    patch = img[40:200, 40:200, 30:120]

    return patch



def preprocess_label(img):
    ch0 = img == 0
    ch1 = img == 1
    ch2 = img == 2
    ch4 = img == 4
    print("Channel shape: ", ch0.shape)
    ch0 = ch0[40:200, 40:200, 30:120]
    ch1 = ch1[40:200, 40:200, 30:120]
    ch2 = ch2[40:200, 40:200, 30:120]
    ch4 = ch4[40:200, 40:200, 30:120]

    output = np.array([ch0, ch2], dtype=np.uint8)

    return output

