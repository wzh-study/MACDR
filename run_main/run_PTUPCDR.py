import torch
import tqdm
from time import time
from data_process.Load_data import Datasets
from models.PTUPCDR import *
from utils.log import Logger
from utils.set import *

class Run(Datasets):
    def __init__(self, 
                 config):
        super(Run, self).__init__(config)

        self.config = config
        self.runid = config['runid']
        self.pre_train_epoch = config['pre_train_epoch']
        self.epoch = config['epoch']  #  10
        self.latent_dim = config['latent_dim']  #  32
        self.meta_dim = config['meta_dim']  #  10
        self.num_fields = config['num_fields']  #  2
        self.lr = config['lr']  #  0.001
        self.lr_meta = config['lr_meta']  #  0.001
        
        self.weight_decay = config['weight_decay']  #  0
        self.config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        record_path = './saved/' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + '/tgt_' + self.tgt + '_src_' + self.src + '_results/PTUPCDR/'+'runid_'+str(self.runid)+'/'
        self.model_save_path = record_path + 'models/'
        self.model_save_pre_train_path = record_path + 'pretrain_models/'

        self.best_epoch = 0
        self.early_stop = 0
        self.results = {'tgt_mae': 10, 'tgt_rmse': 10, 'ptupcdr_mae': 10, 'ptupcdr_rmse': 10}
        
        ensureDir(record_path)
        ensureDir(self.model_save_path)
        ensureDir(self.model_save_pre_train_path)

        self.log = Logger(record_path)
        for key in config.keys():
            self.log.write(set_color(key + '=' + str(config[key]) + '\n', 'yellow'))
    
    def get_data(self):
        print('========Reading data========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)  #  <torch.utils.data.dataloader.DataLoader object at 0x7fa1c76e9f10>
        print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))  #  src 6631 iter / batchsize = 256 
        #self.adj_matrix_s = self._lightgcn_adj_matrix(self.src_path)

        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))
        #self.adj_matrix_t = self._lightgcn_adj_matrix(self.tgt_path)
        
        #  PUTUCDR
        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta, history=True)
        print('meta {} iter / batchsize = {} '.format(len(data_meta), self.batchsize_meta))  #  meta 2924 iter / batchsize = 128

    
        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))  #  test 734 iter / batchsize = 128

        return data_src, data_tgt, data_meta, data_test

    def get_model(self):
        if self.base_model == 'MF':
            pre_train_model = Pre_Train(self.config, self.uid_s, self.iid_s, self.uid_t, self.iid_t)
            
        else:
            raise ValueError('Unknown base model: ' + self.base_model)

        model = PTUPCDR(self.config)  #  (181187, 114495, 2, 10, 50)
        if self.use_cuda:
            pre_train_model = pre_train_model.cuda()
            model = model.cuda()
        return pre_train_model, model

    def get_optimizer(self, pre_train_model):
        optimizer_src = torch.optim.Adam(params=pre_train_model.src_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)  #  self.wd=0
        optimizer_tgt = torch.optim.Adam(params=pre_train_model.tgt_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        print('\n pre-train parameters')
        for name, param in pre_train_model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        return optimizer_src, optimizer_tgt
        # optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)  #  self.weight_decay=0
        # optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optimizer_meta = torch.optim.Adam(params=[{'params': model.meta_net.parameters()}, {'params': model.tgt_model.parameters()}], lr=self.lr_meta, weight_decay=self.weight_decay)
        
        # return optimizer_src, optimizer_tgt, optimizer_meta

    def eval_mae(self, model, data_loader, stage):
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                pred = model(X, stage)  #  torch.Size([128])
                targets.extend(y.squeeze(1).tolist())  #  每次128个评分   len=128
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()  #  torch.Size([93866])
        predicts = torch.tensor(predicts)  #  torch.Size([93866])

        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()

    def train(self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False):
        print('Training Epoch {}:'.format(epoch + 1))
        model.train()
        for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            if mapping:  #  EMCDR
                src_emb, tgt_emb = model(X, stage)
                loss = criterion(src_emb, tgt_emb)
            else:
                pred = model(X, stage)
                loss = criterion(pred, y.squeeze().float())  #  与真实评分求分数差
            model.zero_grad()
            loss.backward()
            optimizer.step()

    def update_results(self, model, epoch, mae, rmse, phase, model_save_path):
        if mae < self.results[phase + '_mae'] and rmse < self.results[phase + '_rmse']:
            self.early_stop = 0
            self.results[phase + '_mae'] = mae
            self.results[phase + '_rmse'] = rmse
            self.best_epoch = epoch
            self.best_ckpt = 'epoch_'+ str(epoch) + '_mae_' + str(self.results[phase + '_mae']) + '.ckpt'
            save_checkpoint(model, model_save_path, self.best_ckpt, max_checkpoints=1)

        elif mae < self.results[phase + '_mae']:
            self.early_stop = 0
            self.results[phase + '_mae'] = mae
            self.results[phase + '_rmse'] = rmse
            self.best_epoch = epoch
            self.best_ckpt = 'epoch_'+ str(epoch) + '_mae_' + str(self.results[phase + '_mae']) + '.ckpt'
            save_checkpoint(model, model_save_path, self.best_ckpt, max_checkpoints=1)

        else:
            self.early_stop += 1

    
    def CDR(self, pre_train_model, model, data_src, data_tgt, data_meta, data_test,
            criterion, optimizer_src, optimizer_tgt):
        t1 = time()
        print('=====CDR Pretraining=====')
        if len(os.listdir(self.model_save_pre_train_path)) > 0:
            file_names = os.listdir(self.model_save_pre_train_path)
            checkpoint = torch.load(self.model_save_pre_train_path + file_names[0])
            pre_train_model.load_state_dict(checkpoint)
        
        else:
            print('Src Pretraining')
            for i in range(self.pre_train_epoch):
                self.train(data_src, pre_train_model, criterion, optimizer_src, i, stage='train_src')
            print('Tgt Pretraining')
            for i in range(self.pre_train_epoch):
                self.train(data_tgt, pre_train_model, criterion, optimizer_tgt, i, stage='train_tgt')
                mae, rmse = self.eval_mae(pre_train_model, data_test, stage='test_tgt')
                self.update_results(pre_train_model, i, mae, rmse, 'tgt', self.model_save_pre_train_path)
                if i >= 10  and self.early_stop >= 5:
                    break  
                print('MAE: {} RMSE: {}'.format(mae, rmse))
        
        
        self.early_stop = 0  #  早停归0

        model.src_model = pre_train_model.src_model
        model.tgt_model = pre_train_model.tgt_model

        optimizer_meta = torch.optim.Adam(params=model.parameters(), lr=self.lr_meta, weight_decay=self.weight_decay)
        #  源域参数不更新为False
        for param in model.src_model.parameters():
            param.requires_grad = False
        # for param in model.src_user_cluster_center.parameters():
        #     param.requires_grad = False
        
        print('\n model parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    
        t2 = time()
    
        print('==========PTUPCDR==========')
        for i in range(self.epoch):  
            self.train(data_meta, model, criterion, optimizer_meta, i, stage='train_meta')
            mae, rmse = self.eval_mae(model, data_test, stage='test_meta')
            self.update_results(model, i, mae, rmse, 'ptupcdr', self.model_save_path)
            self.log.write(set_color('Epoch:{:d}, Test_tgt_MAE:{:.5f}, Test_tgt_RMSE:{:.5f}\n\n'
              .format(i, mae, rmse), 'blue'))
            if i >= 10 and self.early_stop >= 10:
                self.log.write(set_color('Early_stop_Epoch:{:d}\n'
              .format(i), 'blue'))
                break  
        t3 = time()
        self.log.write(set_color('Pre_train_Time:{:.5f}, Train_time:{:.5f}\n'
              .format((t2 - t1), (t3 - t2)), 'blue'))

    def main(self):
        data_src, data_tgt, data_meta, data_test = self.get_data()  #  得到数据
        pre_train_model, model = self.get_model()
        optimizer_src, optimizer_tgt = self.get_optimizer(pre_train_model)
        criterion = torch.nn.MSELoss()
        

        self.CDR(pre_train_model, model, data_src, data_tgt, data_meta, data_test,
                 criterion, optimizer_src, optimizer_tgt)  #  跨域推荐
        
        self.log.write(set_color('PTUPCDR: Best_Epoch:{:d}, Best_Tgt_MAE:{:.5f}, Best_Tgt_RMSE:{:.5f}\n'
              .format(self.best_epoch, self.results['ptupcdr_mae'], self.results['ptupcdr_rmse']), 'blue'))
