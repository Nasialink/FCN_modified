import os
import json
import torch
import loader
import datetime
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import DiceLoss
from model import DiceScore
from torchmetrics.classification import Dice
from model import BrainTumorSegmentationModel


from scores import generate_figs





d = str(datetime.datetime.now())
d = d.replace(" ", "__")
d = d.replace(":", "_")
d = d.replace("-", "_")
d = d.split('.')[0]

exp = "./exp_" + str(d)
exp_inf = exp + "/inf"
exp_metrics = exp + "/metrics"
exp_figures = exp + "/figures"

config = {
    "experiment_folder": exp,
    "experiment_inference_folder": exp_inf,
    "experiment_metrics_folder": exp_figures,
    "batch_size": 2,
    "epochs": 4,
    "learning_rate": 0.00001,
    "classes": 4
}

try:  
    os.mkdir(exp)
    os.mkdir(exp_inf)
    os.mkdir(exp_figures)
    os.mkdir(exp_metrics) 
except OSError as error:  
    print(error)  

dataset = np.load('/home/superteam/test/dataset.npy')

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

batch_size = config["batch_size"]

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
output_channels = config["classes"]

model = BrainTumorSegmentationModel(input_shape, output_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
lambda1 = lambda epoch: (1-(epoch/300))**0.9
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
criterion = DiceLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model.to(device)
torch.cuda.empty_cache()
print("Cuda available: ", torch.cuda.is_available())

epochs = config["epochs"]

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

dice_p = Dice(average='macro', num_classes=config["classes"])
# dice_p = DiceScore(num_classes=4)
dice_p.to(device)
for epoch in tqdm(range(epochs)):
    # print("Epoch: ", epoch)
    model.train(True)
    current_loss = 0.0
    step = 0
    dice_p.reset()
    for x, y in train_ldr:
        x, y = prepare_data(device, x, y)
        optimizer.zero_grad()
        out_gt = model(x)
        loss_gt_value = criterion(out_gt, y)
        loss_gt_value.backward()

        dice_p.update(out_gt, y)
        optimizer.step()
        current_loss  += loss_gt_value * batch_size
        step += 1

    epoch_score = dice_p.compute()
    epoch_loss  = current_loss / len(train_ldr.dataset)
    train_dice[epoch] = epoch_score.item()
    train_loss[epoch] = epoch_loss.item()
    print("Learning rate: ", optimizer.param_groups[0]['lr'])
    # scheduler.step()

    model.train(False)
    step = 0
    dice_p.reset()
    for x, y in valid_ldr:
        x, y = prepare_data(device, x, y)
        
        with torch.no_grad():
            out_gt = model(x)

        out_gt = torch.argmax(out_gt, dim=1)
        dice_p.update(out_gt, y)

        step += 1

    epoch_score = dice_p.compute()
    epoch_loss  = current_loss / len(train_ldr.dataset)
    valid_dice[epoch] = epoch_score.item()
    valid_dice[epoch] = epoch_score.item()

    if epoch >= 0 and epoch_score > max_score:
        print("Max score epoch, score: ", epoch, epoch_score)
        max_score = epoch_score
        model_dict = model.state_dict()
        torch.save(model_dict, exp + '/best_model.pth')


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
    plt.savefig(exp_inf + '/inf_' + str(s) + '.png')
    plt.close()
    s += 1

# test_set_score = self.metrics.f1.compute()
test_set_score = dice_p.compute()

print(test_set_score)
f = open(exp + "/test_set_score.txt", "w")
f.write("Score: " + str(test_set_score.item()))
f.close()

np.save(exp_metrics + '/train_dice', train_dice)
np.save(exp_metrics + '/train_loss', train_loss)
np.save(exp_metrics + '/valid_dice', valid_dice)


generate_figs(exp)
with open(exp + '/config.json', 'w') as fp:
    json.dump(config, fp)

