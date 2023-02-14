import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from model_DeepSense import ActivityClassifier_DPS
from model_HMGAN import G_conv, D_conv
from get_data import get_data
import torchmetrics
from metrics import get_predictive_score
from metrics import get_discriminative_score
from metrics import get_TSTR_score
from sklearn.model_selection import train_test_split

class DASolver_HMGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.N_epochs_GAN = args.N_epochs_GAN
        self.N_epochs_ALL = args.N_epochs_ALL
        self.N_steps_D = args.N_steps_D
        self.N_epochs_C = args.N_epochs_C
        self.N_epochs_DA = args.N_epochs_DA
        self.lr_G = args.lr_G
        self.lr_D = args.lr_D
        self.lr_C = args.lr_C

        self.latent_dim = args.latent_dim
        self.w_mg = args.w_mg
        self.w_gp = args.w_gp
        self.w_mod = args.w_mod
        self.w_gc = args.w_gc

        self.train_loader, self.valid_loader, self.test_loader = get_data(args)

        self.to_save = args.to_save
        self.dataset = args.dataset
        self.tag = args.dataset + '_' + args.model_type + '_fold' + str(args.test_fold)

        self.model_path = args.data_dir + 'checkpoints/' + self.tag
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.N_channels_per_mod = args.N_channels_per_mod
        if 'UTD_MHAD' in args.dataset:
            self.N_modalities = args.N_modalities_UM
            self.N_channels = args.N_channels_UM
            self.N_classes = args.N_classes_UM_arm
            self.seq_len = args.window_UM
            self.N_intervals = args.N_intervals_UM
            self.len_intervals = int(args.window_UM / args.N_intervals_UM)
        elif args.dataset == 'OPPORTUNITY':
            self.N_modalities = args.N_modalities_O
            self.N_channels = args.N_channels_O
            self.N_classes = args.N_classes_O
            self.seq_len = args.window_O
            self.N_intervals = args.N_intervals_O
            self.len_intervals = int(args.window_O / args.N_intervals_O)
        elif args.dataset == 'UCI_HAR':
            self.N_modalities = args.N_modalities_U
            self.N_channels = args.N_channels_U
            self.N_classes = args.N_classes_U
            self.seq_len = args.window_U
            self.N_intervals = args.N_intervals_U
            self.len_intervals = int(args.window_U / args.N_intervals_U)

        self.G = G_conv(self.latent_dim+self.N_classes, self.seq_len, self.N_modalities, args.N_channels_per_mod)
        self.D = D_conv(args.N_channels_per_mod+self.N_classes, self.seq_len, self.N_modalities, self.N_channels_per_mod)
        self.C = ActivityClassifier_DPS(self.N_modalities, self.N_classes, self.N_intervals, self.len_intervals, args.p_drop)

        self.G.cuda()
        self.D.cuda()
        self.C.cuda()

        self.opt_g = optim.Adam(self.G.parameters(), lr=args.lr_G, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.D.parameters(), lr=args.lr_D, betas=(0.5, 0.999))
        self.opt_gc = optim.Adam(self.G.parameters(), lr=args.lr_G, betas=(0.5, 0.999))
        self.opt_c = optim.Adam(self.C.parameters(), lr=args.lr_C)

    def reset_grad(self):
        self.opt_gc.zero_grad()
        self.opt_g.zero_grad()
        self.opt_d.zero_grad()
        self.opt_c.zero_grad()

    def sample_z(self):
        z = Variable(torch.randn((self.batch_size, self.latent_dim), dtype=torch.float32).cuda())
        return z

    def get_D_loss(self, logits_d_mod_r, logits_d_glb_r, logits_d_mod_g, logits_d_glb_g, x_r, x_g, y_inter):
        eps = torch.zeros(self.args.batch_size, 1, 1).uniform_().cuda()
        x_inter = [eps * x_r[i] + (1 - eps) * x_g[i] for i in range(self.N_modalities)]
        logits_d_mod_inter, logits_d_glb_inter = self.D(x_inter, y_inter)
        d_loss_mod = [self.modal_D_loss(logits_d_mod_r[i], logits_d_mod_g[i], x_inter[i], logits_d_mod_inter[i]) for i in range(self.N_modalities)]
        d_loss_glb = self.global_D_loss(logits_d_glb_r, logits_d_glb_g, x_inter, logits_d_glb_inter)
        d_loss_mod_sum = sum([d_loss_mod[i] * self.w_mod[i] for i in range(self.N_modalities)])
        d_loss = d_loss_glb * self.w_mg + d_loss_mod_sum * (1 - self.w_mg)
        return d_loss

    def get_G_loss(self, logits_d_mod_g, logits_d_glb_g):
        g_loss_mod = [self.single_G_loss(logits_d_mod_g[i]) for i in range(self.N_modalities)]
        g_loss_glb = self.single_G_loss(logits_d_glb_g)
        g_loss_mod_sum = sum([g_loss_mod[i] * self.w_mod[i] for i in range(self.N_modalities)])
        g_loss = g_loss_glb * self.w_mg + g_loss_mod_sum * (1 - self.w_mg)
        return g_loss     

    def modal_D_loss(self, logits_d_r, logits_d_g, x_inter, logits_inter):
        grads = autograd.grad(outputs=logits_inter, inputs=x_inter,
                              grad_outputs=torch.ones_like(logits_inter),
                              create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
        grad_pen = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()

        d_loss = -logits_d_r.mean() + logits_d_g.mean() + self.w_gp * grad_pen
        return d_loss

    def global_D_loss(self, logits_d_r, logits_d_g, x_inter, logits_inter):
        grads = [autograd.grad(outputs=logits_inter, inputs=x_inter[i],
                              grad_outputs=torch.ones_like(logits_inter),
                              create_graph=True, retain_graph=True,
                              only_inputs=True)[0] for i in range(self.N_modalities)]
        grads = torch.cat(grads, dim=1)
        grad_pen = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()

        d_loss = -logits_d_r.mean() + logits_d_g.mean() + self.w_gp * grad_pen
        return d_loss

    def single_G_loss(self, logits_d_g):
        g_loss = -logits_d_g.mean()
        return g_loss

    def forward_pass(self, x_r, y_r, type):
        z_g = self.sample_z()
        x_g = self.G(z_g, y_r)
        if type == 'get_x_g':
            return x_g

        if type != 'train_C':
            logits_d_mod_g, logits_d_glb_g = self.D(x_g, y_r)
            if 'train_G' not in type:
                logits_d_mod_r, logits_d_glb_r = self.D(x_r, y_r)
        
        if type != 'train_D':
            if type != 'train_G':
                logits_c_g = self.C(x_g)
            if 'train_G' not in type:
                logits_c_r = self.C(x_r)
        
        if type == 'train_D':
            return logits_d_mod_r, logits_d_glb_r, logits_d_mod_g, logits_d_glb_g, x_g
        elif type == 'train_C':
            return logits_c_r, logits_c_g
        elif type == 'train_GC':
            return logits_d_mod_g, logits_d_glb_g, logits_c_g
        elif type == 'train_G':
            return logits_d_mod_g, logits_d_glb_g

    def train(self):
        self.train_GAN()
        self.train_all()
        test_acc, test_f1 = self.train_C(training=False)
        return test_acc, test_f1

    def train_GAN(self):
        print('\n>>> Start Training GAN...')

        # lossess
        Loss_g = torchmetrics.MeanMetric().cuda()
        Loss_d = torchmetrics.MeanMetric().cuda()

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        for epoch in range(self.N_epochs_GAN):

            self.G.train()
            self.D.train()
            
            for batch_idx, (x_r, y_r) in enumerate(self.train_loader):
                x_r = Variable(x_r.cuda())
                x_r = x_r.permute(0, 2, 1)
                x_r = torch.split(x_r, self.N_channels_per_mod, dim=1)
                y_r = F.one_hot(y_r.long(), num_classes=self.N_classes)
                y_r = Variable(y_r.float().cuda())

                self.reset_grad()

                ''' train discriminator '''
                for _ in range(self.N_steps_D):
                    logits_d_mod_r, logits_d_glb_r, logits_d_mod_g, logits_d_glb_g, x_g = self.forward_pass(x_r, y_r, 'train_D')
                    D_loss = self.get_D_loss(logits_d_mod_r, logits_d_glb_r, logits_d_mod_g, logits_d_glb_g, x_r, x_g, y_r)
                    D_loss.backward()
                    self.opt_d.step()
                    self.reset_grad()

                ''' train generator '''
                G_loss = 0
                for _ in range(2):
                    logits_d_mod_g, logits_d_glb_g = self.forward_pass(x_r, y_r, 'train_G')
                    G_loss += self.get_G_loss(logits_d_mod_g, logits_d_glb_g)
                G_loss.backward()
                self.opt_g.step()
                self.reset_grad()

                # track training losses and metrics after optimization
                Loss_d(D_loss)
                Loss_g(G_loss)

            print('Train Epoch {}: Train: Loss_d:{:.6f} Loss_g:{:.6f}'.format(
                epoch, Loss_d.compute().item(), Loss_g.compute().item()))

            Loss_g.reset()
            Loss_d.reset()

    def train_all(self):
        print('\n>>> Start Training GAN and Classifier...')
        max_tstr_score = 0

        criterion_c = nn.CrossEntropyLoss().cuda()

        # lossess
        Loss_g = torchmetrics.MeanMetric().cuda()
        Loss_d = torchmetrics.MeanMetric().cuda()
        Loss_c = torchmetrics.MeanMetric().cuda()

        # classification accuracies of real and generated data
        train_c_acc_r = torchmetrics.Accuracy().cuda()
        train_c_acc_g = torchmetrics.Accuracy().cuda()

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)        

        for epoch in range(self.N_epochs_ALL):

            self.G.train()
            self.D.train()
            self.C.train()

            for batch_idx, (x_r, y_r) in enumerate(self.train_loader):
                x_r = Variable(x_r.cuda())
                x_r = x_r.permute(0, 2, 1)
                x_r = torch.split(x_r, self.N_channels_per_mod, dim=1)
                y_r = F.one_hot(y_r.long(), num_classes=self.N_classes)
                y_r = Variable(y_r.float().cuda())

                ''' train discriminator '''
                for _ in range(self.N_steps_D):
                    self.reset_grad()
                    logits_d_mod_r, logits_d_glb_r, logits_d_mod_g1, logits_d_glb_g1, x_g = self.forward_pass(x_r, y_r, 'train_D')
                    D_loss = self.get_D_loss(logits_d_mod_r, logits_d_glb_r, logits_d_mod_g1, logits_d_glb_g1, x_r, x_g, y_r)
                    D_loss.backward()
                    self.opt_d.step()

                ''' train classifier '''
                self.reset_grad()
                logits_c_r, logits_c_g = self.forward_pass(x_r, y_r, 'train_C')
                C_loss_r = criterion_c(logits_c_r, y_r)
                if epoch >= self.N_epochs_DA:
                    C_loss_g = criterion_c(logits_c_g, y_r)
                    C_loss = (C_loss_r + C_loss_g) / 2
                    if epoch == self.N_epochs_DA and batch_idx == 0:
                        print('DA!')
                else:
                    C_loss = C_loss_r
                C_loss.backward()
                self.opt_c.step()
                
                ''' train generator '''
                self.reset_grad()
                G_loss_GAN = 0
                G_loss_C = 0
                for _ in range(2):
                    logits_d_mod_g, logits_d_glb_g, logits_c_g = self.forward_pass(x_r, y_r, 'train_GC')
                    G_loss_GAN += self.get_G_loss(logits_d_mod_g, logits_d_glb_g)
                    G_loss_C += criterion_c(logits_c_g, y_r)
                G_loss = G_loss_GAN + self.w_gc * G_loss_C
                G_loss.backward()
                self.opt_gc.step()
                self.reset_grad()

                # track training losses and metrics after optimization
                Loss_d(D_loss)
                Loss_c(C_loss)
                Loss_g(G_loss)
                train_c_acc_r(logits_c_r.softmax(dim=-1), y_r.long())
                train_c_acc_g(logits_c_g.softmax(dim=-1), y_r.long())

            if (epoch+1) % 10 == 0:
                test_tstr_score = self.eval_tstr(training=True)
                if self.to_save and test_tstr_score > max_tstr_score:
                    max_tstr_score = test_tstr_score
                    torch.save(self.G.state_dict(), self.model_path + '/g.pkl')
                    torch.save(self.D.state_dict(), self.model_path + '/d.pkl')
                    print('best tstr model saved!')

            print('Train Epoch {}: Train: c_acc_r:{:.6f} c_acc_f:{:.6f} Loss_d:{:.6f} Loss_c:{:.6f} Loss_g:{:.6f}'.format(
                epoch, train_c_acc_r.compute().item(), train_c_acc_g.compute().item(), Loss_d.compute().item(), Loss_c.compute().item(), Loss_g.compute().item()))

            Loss_g.reset()
            Loss_d.reset()
            Loss_c.reset()
            train_c_acc_r.reset()
            train_c_acc_g.reset()
        
        test_c_acc, test_c_f1 = self.eval_C(training=True, test_loader=self.test_loader)
        test_acc = test_c_acc
        test_f1 = test_c_f1

        print('>>> Training Finished!')
        return test_acc, test_f1      

    def train_C(self, training=False):
        print('\n>>> Start Training Classifier...')

        aug_loader = self.get_gen_dataset(training, type='aug')

        criterion_c = nn.CrossEntropyLoss().cuda()

        train_c_acc = torchmetrics.Accuracy().cuda()
        Loss_c = torchmetrics.MeanMetric().cuda()

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        for epoch in range(self.N_epochs_C):

            self.C.train()

            for batch_idx, (x, y) in enumerate(aug_loader):
                x = Variable(x.cuda())
                x = x.permute(0, 2, 1)
                x = torch.split(x, self.N_channels_per_mod, dim=1)
                y = Variable(y.long().cuda())

                self.reset_grad()

                ''' train classifier '''
                logits_c = self.C(x)

                loss_c = criterion_c(logits_c, y)
                loss_c.backward()
                self.opt_c.step()
                self.reset_grad()

                # track training losses and metrics after optimization
                Loss_c(loss_c)
                train_c_acc(logits_c.softmax(dim=-1), y) 

            print('Train Epoch {}: Train: c_acc:{:.6f} Loss_c:{:.6f}'.format(
                epoch, train_c_acc.compute().item(), Loss_c.compute().item()))

            train_c_acc.reset()   

        test_c_acc, test_c_f1 = self.eval_C(training=True, test_loader=self.test_loader)
        test_acc = test_c_acc
        test_f1 = test_c_f1   

        if self.to_save:
            torch.save(self.C.state_dict(), self.model_path + '/c.pkl')

        print('>>> Training Finished!')
        return test_acc, test_f1

    def eval_C(self, training, test_loader):
        '''
        training==True:  the model is tested during training, use the current model and print test result in training info
        training==False: the model is tested after training, load the saved model and print test result alone
        '''
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        if not training:
            self.C.load_state_dict(torch.load((self.model_path + '/c.pkl')))

        test_c_acc = torchmetrics.Accuracy().cuda()
        test_c_f1 = torchmetrics.F1Score(num_classes=self.N_classes, average='macro').cuda()

        self.C.eval()

        for _, (x, y) in enumerate(test_loader):
            x = Variable(x.cuda())
            x = x.permute(0, 2, 1)
            x = torch.split(x, self.N_channels_per_mod, dim=1)
            y = Variable(y.long().cuda())

            logits_c = self.C(x)

            # track training losses and metrics 
            test_c_acc(logits_c.softmax(dim=-1), y)
            test_c_f1(logits_c.softmax(dim=-1), y)

        if not training:       
            print('\n>>> Start Testing ...') 
            print(self.tag + ' test acc:{:.6f} test f1:{:.6f}'.format(
                test_c_acc.compute().item(), test_c_f1.compute().item())) 
        return test_c_acc.compute().item(), test_c_f1.compute().item()

    def eval_gen_data(self, training=True):
        gen_loader = self.get_gen_dataset(training)

        predictive_score = get_predictive_score(self.args, self.train_loader, gen_loader, self.N_channels)

        train_d_loader, test_d_loader = self.get_disc_dataset(training)
        discriminative_score, disc_acc = get_discriminative_score(self.args, train_d_loader, test_d_loader, self.N_channels)

        tstr_score = get_TSTR_score(self.args, self.train_loader, gen_loader, self.N_modalities, self.N_channels_per_mod, self.N_classes, self.N_intervals, self.len_intervals)

        return predictive_score, discriminative_score, disc_acc, tstr_score

    def eval_tstr(self, training=True):
        gen_loader = self.get_gen_dataset(training)
        tstr_score = get_TSTR_score(self.args, self.train_loader, gen_loader, self.N_modalities, self.N_channels_per_mod, self.N_classes, self.N_intervals, self.len_intervals)
        return tstr_score

    def get_gen_dataset(self, training=False, type='gen'):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        if not training:
            self.G.load_state_dict(torch.load((self.model_path + '/g.pkl')))

        self.G.eval()

        data_g = []
        label_g = []
        if type == 'aug':
            data_r = []
            label_r = []
        for _, (x_r, y_r) in enumerate(self.train_loader):
            if type == 'aug':
                data_r.append(x_r)
                label_r.append(y_r)            
            x_r = Variable(x_r.cuda())
            x_r = x_r.permute(0, 2, 1)
            x_r = torch.split(x_r, self.N_channels_per_mod, dim=1)
            y_g = F.one_hot(y_r.long(), num_classes=self.N_classes)
            y_g = Variable(y_g.float().cuda())

            for _ in range(self.args.N_aug):
                x_g = self.forward_pass(x_r, y_g, 'get_x_g')
                x_g = [x_g_mod.permute(0, 2, 1) for x_g_mod in x_g]
                x_g = torch.cat(x_g, dim=-1)
                data_g.append(x_g.detach().cpu())
                label_g.append(y_r)

        data_g = torch.concat(data_g)
        label_g = torch.concat(label_g)
        if type == 'aug':
            data_r = torch.concat(data_r)
            label_r = torch.concat(label_r)

        if type == 'gen':
            gen_dataset = TensorDataset(data_g, label_g)   
            gen_loader = DataLoader(gen_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)

            return gen_loader
        elif type == 'aug': 
            data_rg = torch.concat([data_r, data_g])
            label_rg = torch.concat([label_r, label_g])

            aug_dataset = TensorDataset(data_rg, label_rg)   
            aug_loader = DataLoader(aug_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)

            return aug_loader            

    def get_disc_dataset(self, training=True):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        if not training:
            self.G.load_state_dict(torch.load((self.model_path + '/g.pkl')))

        self.G.eval()

        data_aug = []
        yd_aug = []
        for _, (x_r, y_r) in enumerate(self.train_loader):
            data_aug.append(x_r)
            x_r = Variable(x_r.cuda())
            x_r = x_r.permute(0, 2, 1)
            x_r = x_r.unsqueeze(-1)
            y_r = F.one_hot(y_r.long(), num_classes=self.N_classes)
            y_r = Variable(y_r.float().cuda())
            yd = torch.concat([torch.ones(self.batch_size), torch.zeros(self.batch_size)])

            x_g = self.forward_pass(x_r, y_r, 'get_x_g')

            x_g = [x_g_mod.permute(0, 2, 1) for x_g_mod in x_g]
            x_g = torch.cat(x_g, dim=-1)
            data_aug.append(x_g.detach().cpu())
            yd_aug.append(yd)
        data_aug = torch.concat(data_aug)
        yd_aug = torch.concat(yd_aug)

        x_train, x_test, yd_train, yd_test = train_test_split(data_aug, yd_aug, train_size = 0.8, random_state = 0)

        train_d_dataset = TensorDataset(x_train, yd_train)   
        train_d_loader= DataLoader(train_d_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)

        test_d_dataset = TensorDataset(x_test, yd_test)   
        test_d_loader= DataLoader(test_d_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)

        return train_d_loader, test_d_loader