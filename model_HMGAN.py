import torch
import torch.nn as nn

class G_conv(nn.Module):
    def __init__(self, in_dim, seq_len, N_modalities, N_channels_per_mod):
        super().__init__()

        self.N_modalities = N_modalities
        if seq_len == 60:
            self.start = 22
        elif seq_len == 100:
            self.start = 32
        elif seq_len == 128:
            self.start = 39

        self.shared_fc = nn.Sequential(
            nn.Linear(in_dim, self.start * 1 * 32, bias=False),
            nn.BatchNorm1d(self.start * 1 * 32, momentum=0.05, affine=True),
            nn.LeakyReLU(0.2),
        )

        self.shared_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32, momentum=0.05, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32, momentum=0.05, affine=True),
            nn.LeakyReLU(0.2),
            nn.UpsamplingNearest2d(scale_factor=(2, 1)),
        )

        self.mod_conv = nn.ModuleList([nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(16, momentum=0.05, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=(3, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(16, momentum=0.05, affine=True),
            nn.LeakyReLU(0.2),
            nn.UpsamplingNearest2d(scale_factor=(2, 1)),

            nn.Conv2d(16, 8, kernel_size=(3, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(8, momentum=0.05, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, N_channels_per_mod, kernel_size=(3, 1), stride=(1, 1), bias=False),
            nn.Tanh(),
        ) for _ in range(self.N_modalities)])

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, z, y):

        z = torch.cat((z, y), dim=1)

        g = self.shared_fc(z)
        g = g.view(-1, 32, self.start, 1)
        g = self.shared_conv(g)

        g = [self.mod_conv[i](g) for i in range(self.N_modalities)]
        g = [g_mod.squeeze(-1) for g_mod in g]
        return g

class D_conv(nn.Module):
    def __init__(self, in_dim, seq_len, N_modalities, N_channels_per_mod):
        super().__init__()
        self.N_modalities = N_modalities
        self.N_channels_per_mod = N_channels_per_mod
        self.kernel_sizes = [11, 11, 7, 7, 5, 5, 3]
        self.padding = [5, 5, 3, 3, 2, 2, 1]
        self.strides = [1, 2, 1, 2, 1, 2, 1]
        self.kernel_num = [32, 32, 64, 64, 128, 128, 128]
        if seq_len == 100:
            feat_dim1 = 13
            feat_dim2 = 4
        elif seq_len == 60:
            feat_dim1 = 8
            feat_dim2 = 2    
        elif seq_len == 128:
            feat_dim1 = 16
            feat_dim2 = 4          

        self.mod_conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_dim, self.kernel_num[0], self.kernel_sizes[0], self.strides[0], self.padding[0], bias=False),
            nn.LeakyReLU(0.2),
        ) for _ in range(self.N_modalities)])

        for m in range(self.N_modalities):
            for i in range(1, len(self.kernel_sizes)):
                self.mod_conv[m].add_module(str(len(self.mod_conv[m])),
                                        nn.Conv1d(self.kernel_num[i - 1], self.kernel_num[i], self.kernel_sizes[i], self.strides[i], self.padding[i], bias=False))
                self.mod_conv[m].add_module(str(len(self.mod_conv[m])),
                                        nn.LeakyReLU(0.2))

        self.mod_out = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_dim1 * self.kernel_num[-1], 1024, bias=False),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 1, bias=False)
            ) for _ in range(self.N_modalities)])

        self.shared_conv = nn.Sequential(
            nn.Conv1d(self.N_modalities * self.kernel_num[-1], 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )

        self.shared_out = nn.Sequential(
            nn.Linear(feat_dim2 * 64, 1024, bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1, bias=False)
        )

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform(module.weight)

    def forward(self, x, y): # a list of [batch_size, channels, time_steps] for each modality
        mod_x = [None for _ in range(self.N_modalities)]
        mod_prob = [None for _ in range(self.N_modalities)]
        for i in range(self.N_modalities):
            mod_x[i] = label_concat(x[i], y)
            mod_x[i] = self.mod_conv[i](mod_x[i])
            mod_prob[i] = torch.flatten(mod_x[i], start_dim=1)
            mod_prob[i] = self.mod_out[i](mod_prob[i])

        glb_x = torch.cat(mod_x, dim=1)
        glb_x = self.shared_conv(glb_x)
        glb_x = torch.flatten(glb_x, start_dim=1)
        glb_prob = self.shared_out(glb_x)

        return mod_prob, glb_prob

def label_concat(x, y): # x [batch_size, channels, time_steps, 1] y [batch_size, num_classes] onehot
    x_shape = list(x.shape)
    label_shape = list(y.shape)
    y = y.view(label_shape[0], label_shape[1], 1)
    label_shape = list(y.shape)
    y = y * torch.ones(label_shape[0], label_shape[1], x_shape[2]).cuda()
    x = torch.cat((x, y), 1)
    return x