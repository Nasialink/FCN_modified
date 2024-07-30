
import numpy as np
import matplotlib.pyplot as plt


def generate_figs(exp):
    # exp = "exp_2024_07_30__11_36_19"
    metrics = exp + "/metrics/"
    figures = exp + "/figures/"
    print(figures)

    td = np.load(metrics + 'train_dice.npy')
    tl = np.load(metrics + 'train_loss.npy')
    vd = np.load(metrics + 'valid_dice.npy')

    print("Shapes: ", td.shape, tl.shape, vd.shape)

    fs = 14

    plt.figure()
    plt.plot(td[:])
    plt.plot(vd[:])
    plt.xlabel("Epochs", fontsize=fs)
    plt.ylabel("Dice score", fontsize=fs)
    plt.legend(["Training", "Validation"], loc="lower right")
    plt.title("Training - Sets: 175, 50, 25 - 300 epochs - Scheduler OFF")
    plt.savefig(figures + "dice_train_valid.png")
    

    plt.figure()
    plt.plot(tl[:])
    plt.xlabel("Epochs", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.title("Training - Sets: 175, 50, 25 - 300 epochs - Scheduler OFF")
    plt.savefig(figures + "loss_train.png")
    plt.close("all")



if __name__ == '__main__':
    generate_figs("exp_2024_07_30__11_36_19")
