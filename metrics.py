import torch
import torch.nn
import torch.nn as nn
from torch import optim
from torch.nn import init
from torch.autograd import Variable
from model_DeepSense import ActivityClassifier_DPS
import torchmetrics
import numpy as np

def init_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv3d):
        init.xavier_uniform_(module.weight, gain=1)

class Predictor(nn.Module):
    def __init__(self, N_channels):
        super(Predictor, self).__init__()

        self.GRU = nn.GRU(input_size=N_channels,
                          hidden_size=120,
                          num_layers=2,
                          batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(120, N_channels),
            nn.Sigmoid()
        )

        self.apply(init_weights)

    def forward(self, x):  # [batch_size, seq_len, channel_nums]
        x, _ = self.GRU(x)
        x = self.out(x)
        return x

def get_predictive_score(args, real_loader, gen_loader, N_channels):
    """
    Args:
      - real_loader: original data [N_samples, channel_nums, seq_len, 1]
      - gen_loader: generated synthetic data [N_samples, channel_nums, seq_len, 1]
    Returns:
      - predictive_score: MAE of the predictions on the original data
    """

    predictor = Predictor(N_channels)
    predictor.cuda()
    opt_p = optim.Adam(predictor.parameters(), lr=args.lr_pred)
    predictor.train()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for _ in range(args.N_epochs_pred):
        for _, (x, _) in enumerate(gen_loader):
            x = x.cuda() # [batch_size, seq_len, channel_nums]
            x1 = x[:, :-1, :]
            x2 = x[:, 1:, :]

            opt_p.zero_grad()

            x_pred = predictor(x1)
            loss = nn.L1Loss()(x_pred, x2)
            loss.backward()
            opt_p.step()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    predictor.eval()
    predictive_score = 0
    for _, (x, _) in enumerate(real_loader):
        x = x.cuda() # [batch_size, seq_len, channel_nums]
        x1 = x[:, :-1, :]
        x2 = x[:, 1:, :]

        x_pred = predictor(x1)
        loss = nn.L1Loss()(x_pred, x2).item()
        predictive_score += loss

    predictive_score /= len(real_loader)

    return predictive_score

class Discriminator(nn.Module):
    def __init__(self, N_channels):
        super(Discriminator, self).__init__()

        N_layers = 2
        self.GRU = nn.GRU(input_size=N_channels,
                          hidden_size=120,
                          num_layers=N_layers,
                          batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(120*N_layers, 1),
            nn.Sigmoid()
        )

        self.apply(init_weights)

    def forward(self, x):  # [batch_size, seq_len, channel_nums]
        _, x = self.GRU(x)
        x = torch.flatten(x.permute(1,0,2), start_dim=1)
        x = self.out(x)
        return x.squeeze()

def get_discriminative_score(args, train_d_loader, test_d_loader, N_channels):
    """
    Args:
      - real_loader: original data [N_samples, channel_nums, seq_len, 1]
      - gen_loader: generated synthetic data [N_samples, channel_nums, seq_len, 1]
    Returns:
      - discriminative_score: 
    """

    D = Discriminator(N_channels)
    D.cuda()
    opt_d = optim.Adam(D.parameters(), lr=args.lr_GAN)
    D.train()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for _ in range(args.N_epochs_disc):
        for _, (x, y_d) in enumerate(train_d_loader):
            x = x.cuda()
            y_d = y_d.cuda()

            opt_d.zero_grad()

            probs_d = D(x)
            D_loss = torch.nn.BCEWithLogitsLoss()(probs_d, y_d)

            D_loss.backward()
            opt_d.step()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    test_d_acc = torchmetrics.Accuracy().cuda()
    D.eval()
    for _, (x, y_d) in enumerate(test_d_loader):
        x = x.cuda()
        y_d = y_d.cuda()
        
        probs_d = D(x)
        test_d_acc(probs_d, y_d.long())

    disc_acc = test_d_acc.compute().item()
    discriminative_score = np.abs(0.5 - disc_acc)

    return discriminative_score, disc_acc

def get_TSTR_score(args, real_loader, gen_loader, N_modalities, N_channels_per_mod, N_classes, N_intervals, len_intervals, CM=False):
    """
    Args:
      - real_loader: original data [N_samples, channel_nums, seq_len, 1]
      - gen_loader: generated synthetic data [N_samples, channel_nums, seq_len, 1]
    Returns:
      - TSTR score: classification accuracy on the original data
    """

    C = ActivityClassifier_DPS(N_modalities, N_classes, N_intervals, len_intervals, 0)
    C.cuda()
    opt_c = optim.Adam(C.parameters(), lr=args.lr_C)
    C.train()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for _ in range(args.N_epochs_C):
        for _, (x, y) in enumerate(gen_loader):
            x = Variable(x.cuda())
            x = x.permute(0, 2, 1)
            x = torch.split(x, N_channels_per_mod, dim=1)
            y = y.long().cuda()

            opt_c.zero_grad()

            logits_c = C(x)
            loss = nn.CrossEntropyLoss()(logits_c, y)
            loss.backward()
            opt_c.step()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    C.eval()
    test_c_acc = torchmetrics.Accuracy().cuda()
    if CM:
        all_y_true = np.empty([0], dtype=np.int)
        all_y_pred = np.empty([0], dtype=np.int)
    for _, (x, y) in enumerate(real_loader):
        x = Variable(x.cuda())
        x = x.permute(0, 2, 1)
        x = torch.split(x, N_channels_per_mod, dim=1)
        y = y.long().cuda()

        logits_c = C(x)
        test_c_acc(logits_c, y)
        if CM:
            y_pred = logits_c.data.max(1)[1]
            all_y_pred = np.concatenate((all_y_pred, y_pred.cpu().numpy()), axis=0)
            all_y_true = np.concatenate((all_y_true, y.cpu().numpy()), axis=0)

    if CM:
        return test_c_acc.compute().item(), all_y_true, all_y_pred
    else:
        return test_c_acc.compute().item()