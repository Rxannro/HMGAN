import os
import shutil
import args_space
import numpy as np
from scipy import stats
from pandas import Series
from sliding_window import sliding_window
from sklearn.model_selection import KFold
import pandas as pd
import scipy.io

def preprocess_UTD_MHAD(window, stride, data_dir, K=5):
    dataset_path    = data_dir + 'UTD_MHAD/'
    N_channels     = 6

    upper_bound_arm = np.array([3.652832, 7.725342, 6.398193, 1000.519084, 606.778626, 1000.519084]) # max
    lower_bound_arm = np.array([-8.0, -8.0, -8.0, -1000.549618, -1000.549618, -741.557252]) # min

    acts_arm = range(1,22)

    if os.path.exists(dataset_path + 'arm/processed_data/'):
        shutil.rmtree(dataset_path + 'arm/processed_data/')
    os.mkdir(dataset_path + 'arm/processed_data/')

    acts = acts_arm

    upper_bound = upper_bound_arm
    lower_bound = lower_bound_arm

    x_all = np.empty([0, window, N_channels], dtype=np.float)
    y_all = np.empty([0], dtype=np.int)
    for user in range(1,9):
        print( "process arm activity data... user{}".format(user-1))
        time_windows    = np.empty([0, window, N_channels], dtype=np.float)
        act_labels      = np.empty([0], dtype=np.int)

        for act in acts:
            for trial in range(1,5):
                file = dataset_path + 'Inertial/a{}_s{}_t{}_inertial.mat'.format(act, user, trial)

                if not os.path.exists(file):
                    continue

                data  = scipy.io.loadmat(file)['d_iner'] # [?, 6] around 150 time steps

                # normalization
                diff = upper_bound - lower_bound
                data = 2 * (data - lower_bound) / diff - 1

                data[ data > 1 ] = 1.0
                data[ data < -1 ] = -1.0

                #sliding window
                data    = sliding_window(data, (window, N_channels), (stride, 1))
                if len(data.shape) == 2:
                    data = data.reshape(1,data.shape[0],data.shape[1])

                act_min = 1
                label   = np.ones(len(data)) * (act-act_min)

                time_windows    = np.concatenate((time_windows, data), axis=0)
                act_labels      = np.concatenate((act_labels, label), axis=0)

        x_all = np.concatenate((x_all, time_windows), axis=0)
        y_all = np.concatenate((y_all, act_labels), axis=0)

        np.save(dataset_path + 'arm/processed_data/features', x_all)
        np.save(dataset_path + 'arm/processed_data/labels', y_all)
        # save the K fold idx
        kf = KFold(n_splits=K, shuffle=True, random_state=0)
        for i, (train_index, test_index) in enumerate(kf.split(x_all)):
            np.save(dataset_path + 'arm/processed_data/' + 'fold{}_train_idx'.format(i), train_index)
            np.save(dataset_path + 'arm/processed_data/' + 'fold{}_test_idx'.format(i), test_index)

def preprocess_OPPORTUNITY(window, overlap, data_dir, K=5):
    dataset_path    = data_dir + 'OPPORTUNITY/'
    N_channels     = 9

    file_list = [   ['S1-Drill.dat',
                    'S1-ADL1.dat',
                    'S1-ADL2.dat',
                    'S1-ADL3.dat',
                    'S1-ADL4.dat',
                    'S1-ADL5.dat'] ,
                    ['S2-Drill.dat',
                    'S2-ADL1.dat',
                    'S2-ADL2.dat',
                    'S2-ADL3.dat',
                    'S2-ADL4.dat',
                    'S2-ADL5.dat'] ,
                    ['S3-Drill.dat',
                    'S3-ADL1.dat',
                    'S3-ADL2.dat',
                    'S3-ADL3.dat',
                    'S3-ADL4.dat',
                    'S3-ADL5.dat'] ,
                    ['S4-Drill.dat',
                    'S4-ADL1.dat',
                    'S4-ADL2.dat',
                    'S4-ADL3.dat',
                    'S4-ADL4.dat',
                    'S4-ADL5.dat'] ]

    upper_bound = np.array([498.0, 1809.0, 1723.1842000000179, 6794.719200000167, 5843.026200000197, 4011.30700000003, 1678.122800000012, 1225.0, 1446.061400000006])# 0.9999 quantile
    lower_bound = np.array([-1435.0, -832.0, -617.0, -2939.0, -1795.0, -2158.0, -660.0, -1096.0, -928.0])# 0.005 quantile

    if os.path.exists( dataset_path + 'processed_data/' ):
        shutil.rmtree( dataset_path + 'processed_data/' )
    os.mkdir( dataset_path + 'processed_data/' )

    time_windows_all = []
    act_labels_all = []
    for usr_idx in range( 4 ):
        
        print( "process data... user{}".format( usr_idx ) )
        time_windows    = np.empty( [0, window, N_channels], dtype=np.float )
        act_labels      = np.empty( [0], dtype=np.int )

        for file_idx in range( len(file_list[0]) ):

            filename = file_list[ usr_idx ][ file_idx ]

            file    = dataset_path + filename
            signals = pd.read_csv(file, delimiter=' ', header=None)

            signals = signals.loc[:, [50, 51, 52, 53, 54, 55, 56, 57, 58, 249]] # RUA acc xyz gyro xyz mag xyz
            signals.dropna(inplace=True)

            data = signals.values[:,:9]
            label = signals.values[:,-1].astype( np.int )

            label[ label == 0 ] = -1

            # ML_Both_Arms
            label[ label == 406516 ] = 0 # Open Door 1 
            label[ label == 406517 ] = 1 # Open Door 2
            label[ label == 404516 ] = 2 # Close Door 1
            label[ label == 404517 ] = 3 # Close Door 2
            label[ label == 406520 ] = 4 # Open Fridge
            label[ label == 404520 ] = 5 # Close Fridge
            label[ label == 406505 ] = 6 # Open Dishwasher
            label[ label == 404505 ] = 7 # Close Dishwasher
            label[ label == 406519 ] = 8 # Open Drawer 1
            label[ label == 404519 ] = 9 # Close Drawer 1
            label[ label == 406511 ] = 10 # Open Drawer 2
            label[ label == 404511 ] = 11 # Close Drawer 2
            label[ label == 406508 ] = 12 # Open Drawer 3
            label[ label == 404508 ] = 13 # Close Drawer 3
            label[ label == 408512 ] = 14 # Clean Table
            label[ label == 407521 ] = 15 # Drink from Cup
            label[ label == 405506 ] = 16 # Toggle Switch

            # fill missing values using Linear Interpolation
            data    = np.array( [Series(i).interpolate(method='linear') for i in data.T] ).T
            data[ np.isnan( data ) ] = 0.

            # normalization
            diff = upper_bound - lower_bound
            data = ( data - lower_bound ) / diff 

            data[ data > 1 ] = 1.0
            data[ data < 0 ] = 0.0

            #sliding window
            data    = sliding_window( data, (window, N_channels), (overlap, 1) )
            label   = sliding_window( label, window, overlap )
            label   = stats.mode( label, axis=1 )[0][:,0]

            #remove non-interested time windows (label==-1)
            invalid_idx = np.nonzero( label < 0 )[0]
            data        = np.delete( data, invalid_idx, axis=0 )
            label       = np.delete( label, invalid_idx, axis=0 )

            time_windows    = np.concatenate( (time_windows, data), axis=0 )
            act_labels      = np.concatenate( (act_labels, label), axis=0 )
        
        time_windows_all.append(time_windows)
        act_labels_all.append(act_labels)

    time_windows_all = np.concatenate(time_windows_all, axis=0)
    act_labels_all = np.concatenate(act_labels_all, axis=0)

    np.save(dataset_path + '/processed_data/features', time_windows_all)
    np.save(dataset_path + '/processed_data/labels', act_labels_all)
    # save the K fold idx
    kf = KFold(n_splits=K, shuffle=True, random_state=0)
    for i, (train_index, test_index) in enumerate(kf.split(time_windows_all)):
        np.save(dataset_path + '/processed_data/' + 'fold{}_train_idx'.format(i), train_index)
        np.save(dataset_path + '/processed_data/' + 'fold{}_test_idx'.format(i), test_index)

def load_data_UCIHAR(dataset_path, file_list, type):
    x_data_list = []
    for item in file_list:
        item_data = np.array(pd.read_csv(dataset_path + type + '/Inertial Signals/' + item + type + '.txt', delim_whitespace=True, header=None))
        x_data_list.append(item_data)
    x = np.stack(x_data_list, -1)

    y = np.array(pd.read_csv(dataset_path + type + '/y_'+ type + '.txt', names=['Activity'], squeeze=True))
    return x, y

def preprocess_UCIHAR(data_dir, K=5):
    dataset_path    = data_dir + 'UCI_HAR/'

    # get the data from txt files to pandas dataffame
    file_list = ['body_acc_x_', 'body_acc_y_', 'body_acc_z_', 'body_gyro_x_', 'body_gyro_y_', 'body_gyro_z_']
    x_train, y_train = load_data_UCIHAR(dataset_path, file_list, 'train')
    x_test, y_test = load_data_UCIHAR(dataset_path, file_list, 'test')

    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # the data are already preprocessed and filtered, no missing values np.isnan(x).sum()=0, since the data is already preprocessed, we use min max value here
    lower_bound = np.array([-0.7270811, -0.8285408496200001, -0.72422586782, -2.5482600237, -2.3043757869, -1.698266])
    upper_bound = np.array([1.072854472100006, 0.620366, 0.6387655, 2.643864, 3.4056461708005163, 1.60952])
    diff = upper_bound - lower_bound

    if os.path.exists( dataset_path + 'processed_data/' ):
        shutil.rmtree( dataset_path + 'processed_data/' )
    os.mkdir( dataset_path + 'processed_data/' )    

    x = 2 * (x - lower_bound) / diff - 1 # need to keep the last dimension as the channel dimension for bradcasted deduction

    x[ x > 1 ] = 1.0
    x[ x < -1 ] = -1.0

    y = y - 1 # pytorch requires labels to lie within [0, C) 

    np.save(dataset_path + '/processed_data/features', x)
    np.save(dataset_path + '/processed_data/labels', y)
    # save the K fold idx
    kf = KFold(n_splits=K, shuffle=True, random_state=0)
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        np.save(dataset_path + '/processed_data/' + 'fold{}_train_idx'.format(i), train_index)
        np.save(dataset_path + '/processed_data/' + 'fold{}_test_idx'.format(i), test_index)


    


    
    




