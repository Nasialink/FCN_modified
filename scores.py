import numpy as np
import matplotlib.pyplot as plt


td = np.load('train_dice.npy')
tl = np.load('train_loss.npy')
vd = np.load('valid_dice.npy')

print("Shapes: ", td.shape, tl.shape, vd.shape)


plt.figure()
plt.plot(td[:50])
plt.plot(vd[:50])
plt.savefig("test.png")
plt.close()

plt.figure()
plt.plot(tl[:50])
plt.savefig("test1.png")
plt.close()
