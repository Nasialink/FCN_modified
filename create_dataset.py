import os

import math
import matplotlib.pyplot as plt
import numpy as np  # For data ma

import random

dataset = np.load('../datasets/dataset.npy')

print(dataset.shape)


for i in range(10):
    index = random.randint(0, dataset.shape[0])
    mod1 = dataset[index, 0, :, :, 45]
    mod2 = dataset[index, 1, :, :, 45]
    mod3 = dataset[index, 2, :, :, 45]
    mod4 = dataset[index, 3, :, :, 45]
    mask = dataset[index, 4, :, :, 45]

    plt.figure()
    plt.imshow(mod1, cmap='gray')
    plt.savefig('./visualizations/p_' + str(index) + '_mod_1.png')
    plt.close()

    plt.figure()
    plt.imshow(mod2, cmap='gray')
    plt.savefig('./visualizations/p_' + str(index) + '_mod_2.png')
    plt.close()
    
    plt.figure()
    plt.imshow(mod3, cmap='gray')
    plt.savefig('./visualizations/p_' + str(index) + '_mod_3.png')
    plt.close()

    plt.figure()
    plt.imshow(mod4, cmap='gray')
    plt.savefig('./visualizations/p_' + str(index) + '_mod_4.png')
    plt.close()

    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.savefig('./visualizations/p_' + str(index) + '_mask.png')
    plt.close()