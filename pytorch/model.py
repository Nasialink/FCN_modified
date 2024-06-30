import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import urllib.request

from pytorch.group_norm import GroupNormalization as GroupNorm

# # Download GroupNormalization if not already available
# try:
#     from torch_groupnorm import GroupNorm
# except ImportError:
#     print('Downloading torch_groupnorm.py in the current directory...')
#     url = 'https://raw.githubusercontent.com/lukemelas/PyTorch-GroupNorm/master/torch_groupnorm.py'
#     urllib.request.urlretrieve(url, "torch_groupnorm.py")
#     from torch_groupnorm import GroupNorm

class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GreenBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = GroupNorm(num_groups=8, num_channels=out_channels)
        self.gn2 = GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        res = self.res_conv(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += res
        return x

class VAE(nn.Module):
    def __init__(self, input_shape):
        super(VAE, self).__init__()
        c, H, W, D = input_shape
        self.flatten_size = c * H * W * D // (8 ** 3)
        self.encoder = nn.Sequential(
            nn.Conv3d(c, 32, kernel_size=3, padding=1),
            nn.SpatialDropout3d(0.2),
            GreenBlock(32, 32),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            GreenBlock(64, 64),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            GreenBlock(128, 128),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            GreenBlock(256, 256),
            GreenBlock(256, 256),
            GreenBlock(256, 256)
        )

        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.z_mean = nn.Linear(256, 128)
        self.z_var = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, self.flatten_size)

        self.decoder = nn.Sequential(
            GreenBlock(256, 256),
            nn.Conv3d(256, 128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            GreenBlock(128, 128),
            nn.Conv3d(128, 64, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            GreenBlock(64, 64),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            GreenBlock(32, 32),
            nn.Conv3d(32, c, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        z_mean = self.z_mean(x)
        z_var = self.z_var(x)
        std = torch.exp(0.5 * z_var)
        eps = torch.randn_like(std)
        z = z_mean + std * eps
        x = self.fc2(z)
        x = x.view(x.size(0), 256, 2, 2, 2)
        x = self.decoder(x)
        return x, z_mean, z_var

class SegmentationModel(nn.Module):
    def __init__(self, input_shape, output_channels):
        super(SegmentationModel, self).__init__()
        c, H, W, D = input_shape
        self.encoder = nn.Sequential(
            nn.Conv3d(c, 32, kernel_size=3, padding=1),
            nn.SpatialDropout3d(0.2),
            GreenBlock(32, 32),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            GreenBlock(64, 64),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            GreenBlock(128, 128),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            GreenBlock(256, 256),
            GreenBlock(256, 256),
            GreenBlock(256, 256)
        )

        self.decoder_gt = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            GreenBlock(128, 128),
            nn.Conv3d(128, 64, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            GreenBlock(64, 64),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            GreenBlock(32, 32),
            nn.Conv3d(32, output_channels, kernel_size=1)
        )

    def forward(self, x):
        x_enc = self.encoder(x)
        x_gt = self.decoder_gt(x_enc)
        return x_gt, x_enc

class BrainTumorSegmentationModel(nn.Module):
    def __init__(self, input_shape, output_channels):
        super(BrainTumorSegmentationModel, self).__init__()
        self.segmentation_model = SegmentationModel(input_shape, output_channels)
        self.vae = VAE(input_shape)

    def forward(self, x):
        out_gt, x_enc = self.segmentation_model(x)
        out_vae, z_mean, z_var = self.vae(x_enc)
        return out_gt, out_vae, z_mean, z_var

def dice_coefficient(pred, target):
    smooth = 1.
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def loss_gt(pred, target):
    return 1 - dice_coefficient(pred, target)

def loss_vae(input_shape, z_mean, z_var, pred, target, weight_L2=0.1, weight_KL=0.1):
    c, H, W, D = input_shape
    n = c * H * W * D
    loss_L2 = F.mse_loss(pred, target)
    loss_KL = (1 / n) * torch.sum(torch.exp(z_var) + z_mean**2 - 1. - z_var)
    return weight_L2 * loss_L2 + weight_KL * loss_KL

# input_shape = (4, 160, 192, 128)
# output_channels = 3

# model = BrainTumorSegmentationModel(input_shape, output_channels)
# optimizer = optim.Adam(model.parameters(), lr=0.00001)

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
