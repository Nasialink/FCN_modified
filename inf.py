import torch
import loader
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm
# from model import loss_gt
import matplotlib.pyplot as plt
from model import DiceLoss
# from model import dice_coefficient
from torchmetrics.classification import Dice
from model import BrainTumorSegmentationModel




# Melina
# data_path = '/home/superteam/test/data/data_cropped.nii.gz'
# labels_path = '/home/superteam/test/data/labels_cropped.nii.gz'

# A40
#data_path = '/home/azach/testdir/data/data_cropped.nii.gz'
#labels_path = '/home/azach/testdir/data/labels_cropped.nii.gz'

dataset = np.load('/home/superteam/test/dataset.npy')

# np_data = nib.load(data_path).get_fdata()
# np_labels = nib.load(labels_path).get_fdata()
np_data = dataset[:, :4, :, :, :]
np_labels = dataset[:, 4, :, :, :]

print("Dataset shape: ", np_data.shape, np_labels.shape)

data_len = len(np_data)
train_set_len = int(0.7 * data_len)
valid_set_len = int(0.2 * data_len)
test_set_len = int(0.1 * data_len)

train_end = train_set_len
val_start = train_end
val_end = val_start + valid_set_len
test_start = val_end
test_end = test_start + test_set_len

train_set_x = np_data[:train_end, :, :, :, :]
train_set_y = np_labels[:train_end, :, :, :]
valid_set_x = np_data[val_start:val_end, :, :, :, :]
valid_set_y = np_labels[val_start:val_end, :, :, :]
test_set_x = np_data[test_start:, :, :, :, :]
test_set_y = np_labels[test_start:, :, :, :]

print("Dataset x: ", np_data.shape, np.min(np_data), np.max(np_data))
print("Dataset y: ", np_labels.shape, np.min(np_labels), np.max(np_labels))
print("Sets: ", len(train_set_x), len(valid_set_x), len(test_set_x))

batch_size = 2

train_set      = loader.Dataset(train_set_x, train_set_y)
params         = {'batch_size': batch_size, 'shuffle': True}
train_ldr = torch.utils.data.DataLoader(train_set, **params)
valid_set      = loader.Dataset(valid_set_x, valid_set_y)
params         = {'batch_size': batch_size, 'shuffle': False}
valid_ldr = torch.utils.data.DataLoader(valid_set, **params)
test_set       = loader.Dataset(test_set_x, test_set_y)
params         = {'batch_size': batch_size, 'shuffle': False}
test_ldr  = torch.utils.data.DataLoader(test_set, **params)


input_shape = (4, 160, 160, 90)
output_channels = 4

model = BrainTumorSegmentationModel(input_shape, output_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = DiceLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
model.to(device)
# torch.cuda.empty_cache()
print("Cuda available: ", torch.cuda.is_available())

epochs = 60

def prepare_data(device, x, y):
    # x = torch.unsqueeze(x, 1)

    x = x.to(torch.float32)
    y = y.to(torch.int64)

    x = x.to(device)
    y = y.to(device)

    return x, y


def show_patient(x, y, pred, id):
    plt.figure(figsize=(15, 15)) 
    l = ['T1', 'T1ce', 'T2', 'Flair', 'Seg']
    s = random.randint(30, 65)
    for i in range(1, 5):
        plt.subplot(3, 4, i)
        plt.imshow(x[i-1, :, :, s], cmap='gray')
        plt.axis('off')
        plt.title(l[i-1], fontsize=16)

    l1 = ["Target - " + item for item in l]  
    for i in range(5, 9):
        plt.subplot(3, 4, i)
        plt.imshow(x[i-5, :, :, s], cmap='gray')
        plt.imshow(y[:, :, s], alpha=0.3)
        plt.axis('off')
        plt.title(l1[i-5], fontsize=16)

    l2 = ["Pred - " + item for item in l]  
    for i in range(9, 13):
        plt.subplot(3, 4, i)
        plt.imshow(x[i-9, :, :, s], cmap='gray')
        plt.imshow(pred[:, :, s], alpha=0.3)
        plt.axis('off')
        plt.title(l2[i-9], fontsize=16)
        
    plt.savefig('./inf_scheduler/p_'+str(id)+'.png')
    plt.close()

dice_p = Dice(average='micro')
dice_p.to(device)



model.load_state_dict(torch.load('best_model.pth'))
model.eval()
current_score = 0.0
current_loss = 0.0
dice_p.reset()
s = 0
for x, y in test_ldr:
    x, y = prepare_data(device, x, y)
        
    with torch.no_grad():
        out_gt = model(x)
    out_gt = torch.argmax(out_gt, dim=1)
    dice_p.update(out_gt, y)

    xs = x.cpu().detach().numpy()
    preds = out_gt.cpu().detach().numpy()
    targets = y.cpu().detach().numpy()

    
    show_patient(xs[0, :, :, :, :], targets[0, :, :, :], preds[0, :, :, :], s)
    
    s += 1

# test_set_score = self.metrics.f1.compute()
test_set_score = dice_p.compute()

print(test_set_score)