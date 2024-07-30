import numpy as np
import matplotlib.pyplot as plt


td = np.load('train_dice.npy')
tl = np.load('train_loss.npy')
vd = np.load('valid_dice.npy')

print("Shapes: ", td.shape, tl.shape, vd.shape)

fs = 14

plt.figure()
plt.plot(td[:])
plt.plot(vd[:])
plt.xlabel("Epochs", fontsize=fs)
plt.ylabel("Dice score", fontsize=fs)
plt.legend(["Training", "Validation"], loc="lower right")
plt.title("Training - Sets: 175, 50, 25 - 300 epochs - Scheduler OFF")
plt.savefig("test.png")
plt.close()

plt.figure()
plt.plot(tl[:])
plt.xlabel("Epochs", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.title("Training - Sets: 175, 50, 25 - 300 epochs - Scheduler OFF")
plt.savefig("test1.png")
plt.close()
