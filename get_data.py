import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

def get_data(args):
    train_ratio = 0.8
    if 'UTD_MHAD' in args.dataset:
        path = args.data_dir + 'UTD_MHAD/' + args.dataset[-3:] + '/processed_data'
    else:
        path = args.data_dir + args.dataset + '/processed_data'

    x_all = np.load(path+'/features.npy')
    y_all = np.load(path+'/labels.npy')
    train_idx = np.load(path+'/fold{}_train_idx.npy'.format(args.test_fold))
    test_idx = np.load(path+'/fold{}_test_idx.npy'.format(args.test_fold))
    
    x_train, x_test = x_all[train_idx], x_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size = train_ratio, random_state = 0)

    if 'multiply' in args.aug_type:
        x_train, y_train = augment_to_mutiply(args, x_train, y_train)

    train_dataset = TensorDataset(torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train))   
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    valid_dataset = TensorDataset(torch.from_numpy(x_valid.astype(np.float32)), torch.from_numpy(y_valid))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    test_dataset = TensorDataset(torch.from_numpy(x_test.astype(np.float32)), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    return train_loader, valid_loader, test_loader

def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

def augment_to_mutiply(args, data, labels):# instead of changing the DA function, we could just sample train data ahead according to labels
    np.random.seed(args.seed)
    data_to_aug = np.repeat(data, args.N_aug, axis=0)
    labels_to_aug = np.repeat(labels, args.N_aug, axis=0)
    generated_data = window_slice(data_to_aug)
    return np.concatenate([data, generated_data]), np.concatenate([labels, labels_to_aug])