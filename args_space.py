import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    #parameters w.r.t. datasets
    parser.add_argument('--window_U', type=int, default=128, help='the number of readings in a time window in UCI HAR, this dataset is already partitioned into 2.56-second windows')
    parser.add_argument('--overlap_U', type=int, default=0.5, help='the overlap ratio between time windows in UCI HAR')
    parser.add_argument('--N_classes_U', type=int, default=6, help='the number of activity classes in UCI HAR')
    parser.add_argument('--N_channels_U', type=int, default=6, help='the total number of channels in UCI HAR')
    parser.add_argument('--N_modalities_U', type=int, default=2, help='the number of modalities in total in UCI HAR')
    parser.add_argument('--N_users_U', type=int, default=None, help='the number of users in UCI HAR')
    parser.add_argument('--N_intervals_U', type=int, default=8, help='the number of intervals in each window in UCI HAR')

    parser.add_argument('--window_UM', type=int, default=100, help='the number of readings in a time window in UTD-MHAD, this dataset contains 861 data sequences of around 3 seconds')
    parser.add_argument('--stride_UM', type=int, default=50, help='the number of readings to slide between time windows in UTD-MHAD')
    parser.add_argument('--N_modalities_UM', type=int, default=2, help='the number of sensor modalities in UTD-MHAD')
    parser.add_argument('--N_classes_UM_arm', type=int, default=21, help='the number of activity classes in UTD-MHAD')
    parser.add_argument('--N_channels_UM', type=int, default=6, help='the total number of channels in UTD-MHAD')
    parser.add_argument('--N_users_UM', type=int, default=8, help='the number of users in UTD-MHAD')
    parser.add_argument('--N_intervals_UM', type=int, default=10, help='the number of intervals in a window for UTD-MHAD')

    parser.add_argument('--window_O', type=int, default=60, help='the number of readings in a time window in OPPORTUNITY')
    parser.add_argument('--stride_O', type=int, default=30, help='the number of readings to slide between time windows in OPPORTUNITY')
    parser.add_argument('--N_modalities_O', type=int, default=3, help='the number of sensor modalities in OPPORTUNITY')
    parser.add_argument('--N_classes_O', type=int, default=17, help='the number of activity classes in OPPORTUNITY')
    parser.add_argument('--N_channels_O', type=int, default=9, help='the total number of channels in OPPORTUNITY')
    parser.add_argument('--N_users_O', type=int, default=4, help='the number of users in OPPORTUNITY')
    parser.add_argument('--N_intervals_O', type=int, default=10, help='the number of intervals in a window for OPPORTUNITY')

    #parameters w.r.t. general model settings
    parser.add_argument('--N_aug', type=float, default=1, help='the ratio of the amount of the generated data compared to original data')
    parser.add_argument('--lr_G', type=float, default=1e-4, help='learning rate for Generator')
    parser.add_argument('--lr_D', type=float, default=1e-4, help='learning rate for Discriminator')
    parser.add_argument('--lr_C', type=float, default=1e-3, help='learning rate for Classifier')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--N_epochs_GAN', type=int, default=100, help='the number of epochs for stage 1')
    parser.add_argument('--N_epochs_ALL', type=int, default=200, help='the number of epochs for stage 2')
    parser.add_argument('--N_epochs_C', type=int, default=100, help='the number of epochs for classifier training') 
    parser.add_argument('--N_epochs_DA', type=int, default=0, help='the number of epochs to start using generated data for augmentation')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')

    #parameters w.r.t. model structures and training for HMGAN
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--N_channels_per_mod', type=int, default=3, help='the number of channels for each modality')
    parser.add_argument('--p_drop', type=float, default=0.05, help='the probability of dropping out')
    parser.add_argument('--weight_decay', type=float, default=0, help='the coefficient of weight decay (L2 penalty)')
    parser.add_argument('--aug_type', type=str, default='', help='how to augment training data')
    parser.add_argument('--w_mg', type=float, default=0.3)
    parser.add_argument('--w_mod', type=list, default=[0.5,0.5])
    parser.add_argument('--w_gc', type=float, default=1)
    parser.add_argument('--w_gp', type=float, default=10)
    parser.add_argument('--N_steps_D', type=int, default=5)

    #parameters w.r.t. model structures and training for evaluation metrics
    parser.add_argument('--lr_GAN', type=float, default=1e-4)
    parser.add_argument('--lr_pred', type=float, default=1e-3)
    parser.add_argument('--N_epochs_pred', type=int, default=200)
    parser.add_argument('--N_epochs_disc', type=int, default=100)

    #parameters w.r.t. experiment setups
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_type', type=str, help='the model name')
    parser.add_argument('--N_folds', type=int, default=5, help='the number of folds')
    parser.add_argument('--test_fold', type=int, default=0, help='which fold to test on')
    parser.add_argument('--cuda', type=int, default=-1, help='the cuda device to run on')
    parser.add_argument('--to_save', type=bool, default=False, help='whether to save the model')
    parser.add_argument('--data_dir', type=str)

    args = parser.parse_args()

    return args