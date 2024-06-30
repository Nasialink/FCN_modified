import torch
import loader
import numpy as np
import nibabel as nib
from model import loss_gt
from model import dice_coefficient
from model import BrainTumorSegmentationModel




# Melina
# data_path = '/home/itzo/brain/test/data/data_cropped.nii.gz'
# labels_path = '/home/itzo/brain/test/data/labels_cropped.nii.gz'

# A40
data_path = '/home/azach/testdir/data/data_cropped.nii.gz'
labels_path = '/home/azach/testdir/data/labels_cropped.nii.gz'

np_data = nib.load(data_path).get_fdata()
np_labels = nib.load(labels_path).get_fdata()

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
train_set_y = np_labels[:train_end, :, :, :, :]
valid_set_x = np_data[val_start:val_end, :, :, :, :]
valid_set_y = np_labels[val_start:val_end, :, :, :, :]
test_set_x = np_data[test_start:, :, :, :, :]
test_set_y = np_labels[test_start:, :, :, :, :]

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


input_shape = (4, 160, 192, 128)
output_channels = 3

model = BrainTumorSegmentationModel(input_shape, output_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model.to(device)
torch.cuda.empty_cache()

epochs = 100

def prepare_data(device, x, y):
    # x = torch.unsqueeze(x, 1)

    x = x.to(torch.float32)
    y = y.to(torch.int64)

    x = x.to(device)
    y = y.to(device)

    return x, y

train_loss = np.zeros((100))
train_dice = np.zeros((100))
for epoch in range(epochs):
    print("Epoch: ", epoch)
    model.train(True)
    
    current_loss = 0.0
    current_score = 0.0
    step = 0
    for x, y in train_ldr:
        print("Step: ", step)
        x, y = prepare_data(device, x, y)
        optimizer.zero_grad()
        out_gt = model(x)
        loss_gt_value = loss_gt(out_gt, y)
        loss_gt_value.backward()
        dice = dice_coefficient(out_gt, y)
        optimizer.step()
        current_loss  += loss_gt_value * batch_size
        current_score += dice
        print("Current batch score: ", current_score.item())
        step += 1

    epoch_score = current_score / len(train_ldr.dataset)
    epoch_loss  = current_loss / len(train_ldr.dataset)
    train_dice[epoch] = epoch_score.item()
    train_loss[epoch] = epoch_loss.item()

print(train_dice)
print(train_loss)

# # Example forward pass
# x = torch.randn(1, *input_shape)
# out_gt, out_vae, z_mean, z_var = model(x)
# print(out_gt.shape, out_vae.shape, z_mean.shape, z_var.shape)

# # Example loss calculation
# target_gt = torch.randn_like(out_gt)
# target_vae = torch.randn_like(out_vae)

# loss_gt_value = loss_gt(out_gt, target_gt)
# loss_vae_value = loss_vae(input_shape, z_mean, z_var, out_vae, target_vae)

# total_loss = loss_gt_value + loss_vae_value
# total_loss.backward()
# optimizer.step()