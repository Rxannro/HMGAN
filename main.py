from __future__ import print_function
import args_space
import solver_HMGAN
import preprocess
import numpy as np
import warnings
import os
import time
warnings.filterwarnings("ignore")

def main(args):
    if args.cuda != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    acc, f1 = [np.empty([args.N_folds], dtype=np.float) for _ in range(2)]
    p_score, d_score, tstr_score = [np.empty([args.N_folds], dtype=np.float) for _ in range(3)]
    starttime = time.time()
    for test_fold in range(args.N_folds):
        args.test_fold = test_fold     

        if 'HMGAN' in args.model_type:
            mysolver = solver_HMGAN.DASolver_HMGAN(args)
        print('\n=== ' + args.dataset + '_' + args.model_type + '_fold' + str(args.test_fold) + ' ===')

        test_acc, test_f1 = mysolver.train()

        acc[test_fold] = test_acc
        f1[test_fold] = test_f1

        test_p_score, test_d_score, _, test_tstr_score = mysolver.eval_gen_data(training=True)
        p_score[test_fold] = test_p_score
        d_score[test_fold] = test_d_score
        tstr_score[test_fold] = test_tstr_score

    endtime = time.time()

    print('\n=== ' + args.dataset + '_' + args.model_type + ' ===')
    print('Duration: ', round(endtime - starttime, 2), 'secs')
    print(args)
    print("FINAL VALUE: \nacc: ", np.around(acc, 3), "\nf1: ", np.around(f1, 3))
    print("FINAL AVERAGE: \nacc: ", np.around(np.mean(acc), 3), "\nf1: ", np.around(np.mean(f1), 3))
    print("FINAL STD: \nacc: ", np.around(np.std(acc), 3), "\nf1: ", np.around(np.std(f1), 3))
    print("OTHER FINAL VALUE: \np_score: ", np.around(p_score, 3), "\nd_score: ", np.around(d_score, 3), "\ntstr_score: ", np.around(tstr_score, 3))
    print("OTHER FINAL AVERAGE: \np_score: ", np.around(np.mean(p_score), 3), "\nd_score: ", np.around(np.mean(d_score), 3), "\ntstr_score: ", np.around(np.mean(tstr_score), 3))
    print("OTHER FINAL STD: \np_score: ", np.around(np.std(p_score), 3), "\nd_score: ", np.around(np.std(d_score), 3), "\ntstr_score: ", np.around(np.std(tstr_score), 3))

    return np.mean(acc), np.mean(f1)

if __name__ == '__main__':
    args = args_space.get_args()

    # dataset-specific parameters
    if args.dataset == 'UTD_MHAD_arm':
        args.batch_size = 64
        args.w_mg = 0.9
        args.w_mod = [0.5,0.5]
        preprocess.preprocess_UTD_MHAD(args.window_UM, args.stride_UM, args.data_dir)
    elif args.dataset == 'OPPORTUNITY':
        args.batch_size = 128
        args.w_mg = 0.5
        args.w_mod = [1/3, 1/3, 1/3]
        preprocess.preprocess_OPPORTUNITY(args.window_O, args.stride_O, args.data_dir)
    elif args.dataset == 'UCI_HAR':
        args.batch_size = 128
        args.w_mg = 0.9
        args.w_mod = [0.5,0.5]
        preprocess.preprocess_UCIHAR(args.data_dir)

    # general parameters
    args.N_epochs_GAN = 150
    args.N_epochs_ALL = 100
    args.N_epochs_DA = 30
    args.N_epochs_C = 100
    args.p_drop = 0
    args.lr_G = 0.0007
    args.lr_D = 0.0001
    args.lr_C = 0.001
    args.N_steps_D = 5
    args.latent_dim = 100
    args.w_gp = 10
    args.w_gc = 1.2
    args.N_aug = 1

    args.aug_type = 'multiply'
    args.to_save = True
    
    args.model_type = 'HMGAN'
    main(args)