import torch
import loader
import numpy as np
import nibabel as nib
from tqdm import tqdm
# from model import loss_gt
import matplotlib.pyplot as plt
from model import DiceLoss
from model import DiceScore
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
lambda1 = lambda epoch: (1-(epoch/300))**0.9
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
criterion = DiceLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model.to(device)
torch.cuda.empty_cache()
print("Cuda available: ", torch.cuda.is_available())

epochs = 300

def prepare_data(device, x, y):
    # x = torch.unsqueeze(x, 1)

    x = x.to(torch.float32)
    y = y.to(torch.int64)

    x = x.to(device)
    y = y.to(device)

    return x, y

train_loss = np.zeros((epochs))
train_dice = np.zeros((epochs))
valid_loss = np.zeros((epochs))
valid_dice = np.zeros((epochs))

max_score = 0.0

dice_p = Dice(average='macro', num_classes=4)
# dice_p = DiceScore(num_classes=4)
dice_p.to(device)
for epoch in tqdm(range(epochs)):
    # print("Epoch: ", epoch)
    model.train(True)
    current_loss = 0.0
    step = 0
    dice_p.reset()
    for x, y in train_ldr:
        # print("Step: ", step)
        x, y = prepare_data(device, x, y)
        optimizer.zero_grad()
        out_gt = model(x)
        # out_gt = torch.argmax(out_gt, dim=1)
        # out_gt = torch.tensor(out_gt)
        # loss_gt_value = loss_gt(out_gt, y)
        loss_gt_value = criterion(out_gt, y)
        loss_gt_value.backward()

        
        # y = torch.argmax(y, dim=1)
        dice_p.update(out_gt, y)
        optimizer.step()
        current_loss  += loss_gt_value * batch_size
        # print("Current Dice score: ", dice_p.compute().item())
        step += 1

    epoch_score = dice_p.compute()
    epoch_loss  = current_loss / len(train_ldr.dataset)
    train_dice[epoch] = epoch_score.item()
    train_loss[epoch] = epoch_loss.item()
    print("Learning rate: ", optimizer.param_groups[0]['lr'])
    scheduler.step()

    model.train(False)
    step = 0
    dice_p.reset()
    for x, y in valid_ldr:
        # print("Step: ", step)
        x, y = prepare_data(device, x, y)
        
        with torch.no_grad():
            out_gt = model(x)

        out_gt = torch.argmax(out_gt, dim=1)
        # y = torch.argmax(y, dim=1)
        dice_p.update(out_gt, y)

        # print("Validation - Current Dice score: ", dice_p.compute().item())
        step += 1

    epoch_score = dice_p.compute()
    epoch_loss  = current_loss / len(train_ldr.dataset)
    valid_dice[epoch] = epoch_score.item()
    valid_dice[epoch] = epoch_score.item()

    if epoch > 0 and epoch_score > max_score:
        print("Max score epoch, score: ", epoch, epoch_score)
        max_score = epoch_score
        model_dict = model.state_dict()
        torch.save(model_dict, 'best_model.pth')


    print("Train dice, loss - Valid dice: ", train_dice[epoch], train_loss[epoch], valid_dice[epoch])
    print()



model.load_state_dict(model_dict)
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
    preds = out_gt.cpu().detach().numpy()
    targets = y.cpu().detach().numpy()

    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(targets[0,:,:,45], cmap='gray')
    plt.axis('off')
    plt.title('Target', fontsize=8)
    
    plt.subplot(1, 2, 2)
    plt.imshow(preds[0,:,:,45], cmap='gray')
    plt.axis('off')
    plt.title('Prediction', fontsize=8)
    plt.savefig('./inf/inf_' + str(s) + '.png')
    plt.close()
    s += 1

# test_set_score = self.metrics.f1.compute()
test_set_score = dice_p.compute()

print(test_set_score)
# print(train_dice)
# print(train_loss)
# print(valid_dice)

np.save('train_dice', train_dice)
np.save('train_loss', train_loss)
np.save('valid_dice', valid_dice)

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
