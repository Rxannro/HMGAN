import torch
import torch.nn as nn
from itertools import repeat

class ActivityClassifier_DPS(nn.Module):
    def __init__(self, N_modalities, N_classes, N_intervals, len_intervals, p_drop):
        super(ActivityClassifier_DPS, self).__init__()
        self.N_modalities = N_modalities
        self.N_intervals = N_intervals
        self.len_intervals = len_intervals
        K = 2
        C = 64
        self.C = C

        self.mod_conv = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, C, (1, K*3*2), stride=(1, 2*3), bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(),
            SpatialDropout(drop=p_drop),
            nn.Conv2d(C, C, (1, K), bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(),
            SpatialDropout(drop=p_drop),
            nn.Conv2d(C, C, (1, K), bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(),
            SpatialDropout(drop=p_drop),
        ) for _ in range(N_modalities)])

        self.fuse_conv = nn.Sequential(
            nn.Conv3d(C, C, (1, N_modalities, 2), padding='same', bias=False),
            nn.BatchNorm3d(C),
            nn.ReLU(),
            SpatialDropout(drop=p_drop),
            nn.Conv3d(C, C, (1, N_modalities, 2), padding='same', bias=False),
            nn.BatchNorm3d(C),
            nn.ReLU(),
            SpatialDropout(drop=p_drop),
            nn.Conv3d(C, C, (1, N_modalities, 2), padding='same', bias=False),
            nn.BatchNorm3d(C),
            nn.ReLU(),
            SpatialDropout(drop=p_drop)
        )

        self.GRU = nn.GRU(input_size=C * N_modalities * (len_intervals - 3 * (K - 1)), hidden_size=120, num_layers=2, dropout=p_drop)
        self.dropout = nn.Dropout(p=p_drop)

        self.out = nn.Sequential(
            nn.Linear(120, N_classes)
        )

    def forward(self, x):  # a list of [batch_size, channel_nums, seq_len] for each modality
        x = my_fft_torch(x, self.N_intervals, self.len_intervals)
        batch_size = x[0].size()[0]
        N_intervals = x[0].size()[2]
        x = [self.mod_conv[i](x[i]) for i in range(self.N_modalities)]
        feature_len = x[0].size()[3]
        x = [x_mod.reshape(batch_size, self.C, N_intervals, 1, feature_len) for x_mod in x]
        x = torch.cat(x, dim=3)
        x = self.fuse_conv(x)
        x = x.permute(2, 0, 1, 3, 4).reshape(N_intervals, batch_size, -1)
        x, _ = self.GRU(x)
        x = torch.mean(x, dim=0, keepdim=False)
        x = self.out(self.dropout(x))
        return x

def my_fft_torch(tensor_list, N_intervals, len_intervals):
    fft_tensor_list = []
    batch_size = tensor_list[0].shape[0]
    for tensor in tensor_list: # [batch_size, num_channels, seq_len]
        fft_tensor = tensor.permute(0,2,1).reshape(batch_size, N_intervals, len_intervals, 3)
        fft_tensor = torch.fft.fft(fft_tensor, dim=2)
        fft_tensor = torch.cat([fft_tensor.real, fft_tensor.imag], 3)  # [batch_size, N_intervals, interval_length, 3*2] last dimension: real(xyz), imag(xyz)
        fft_tensor = fft_tensor.reshape(batch_size, N_intervals, -1).unsqueeze(1)  # [batch_size, 1, N_intervals, interval_length*3*2] last dim: real(xyz), imag(xyz) at t0, t1, ...

        fft_tensor_list.append(fft_tensor)
    return fft_tensor_list

class SpatialDropout(nn.Module):
    def __init__(self, drop=0.5, noise_shape=None):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        self.noise_shape = noise_shape
        
    def forward(self, inputs):
        """
        inputs: tensor [batch_size, num_channels, ...]
        noise_shape: [batch_size, ...] same dimension as inputs, dropout along the dimensions of value 1
        """
        outputs = inputs.clone()
        if self.noise_shape is None:
            self.noise_shape = (inputs.shape[0], inputs.shape[1], *repeat(1, inputs.dim()-2))  # default: dropout on channel dimension, along all other dimensions
        
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)