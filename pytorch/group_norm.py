import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNormalization(nn.Module):
    def __init__(self, num_groups=32, num_channels=None, eps=1e-5, affine=True):
        super(GroupNormalization, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        N, C, *dims = x.size()
        G = self.num_groups
        if self.num_channels is not None:
            assert C == self.num_channels, 'Expected input with {} channels, but got {} channels'.format(self.num_channels, C)
        assert C % G == 0, 'Number of channels ({}) must be divisible by number of groups ({})'.format(C, G)

        x = x.view(N, G, C // G, *dims)
        mean = x.mean(dim=[2, *range(3, x.ndim)], keepdim=True)
        var = x.var(dim=[2, *range(3, x.ndim)], keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(N, C, *dims)

        if self.affine:
            weight = self.weight.view(1, C, *([1] * (x.ndim - 2)))
            bias = self.bias.view(1, C, *([1] * (x.ndim - 2)))
            x = x * weight + bias
        return x

# Example usage:
if __name__ == '__main__':
    input_tensor = torch.randn(100, 4, 32, 32)
    gn_layer = GroupNormalization(num_groups=2, num_channels=4, eps=0.1)
    output = gn_layer(input_tensor)
    print(output.size())
