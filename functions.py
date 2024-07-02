
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def read_img(img_path):
    return nib.load(img_path).get_fdata()


def preprocess(img):

    # print(img.shape)
    patch = img[40:200, 40:200, 30:120]

    # plt.figure()
    # plt.imshow(patch[:, :, 50], cmap='gray')
    # plt.savefig("xaxa.png")
    # plt.close()
    # print(patch.shape)

    return patch



def preprocess_label(img):
    # patch = img[40:200, 40:200, 30:120]
    # print(img.shape, np.unique(img))
    ch0 = img == 0
    ch1 = img == 1
    ch2 = img == 2
    ch4 = img == 4
    print("Channel shape: ", ch0.shape)
    ch0 = ch0[40:200, 40:200, 30:120]
    ch1 = ch1[40:200, 40:200, 30:120]
    ch2 = ch2[40:200, 40:200, 30:120]
    ch4 = ch4[40:200, 40:200, 30:120]
    # print(np.unique(patch))
    # patch[ch4] = 3
    # print(np.unique(patch))
    output = np.array([ch0, ch2], dtype=np.uint8)
    # print("Output shape: ", output.shape)
    # plt.figure()
    # plt.imshow(ch2[:, :, 50], cmap='gray')
    # plt.savefig("xaxa1.png")
    # plt.close()

    return output

