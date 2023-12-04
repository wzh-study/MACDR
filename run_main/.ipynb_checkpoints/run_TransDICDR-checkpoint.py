import torch
import tqdm
from time import time
from data_process.Load_data import Datasets
from models.TransDICDR import TransDICDR
from utils.log import Logger
from utils.set import *

class Run(Datasets):
    def __init__(self, 
                 config):
        super(Run, self).__init__(config)

        self.runid = config['runid']
        self.pre_train_epoch = config['pre_train_epoch']
        self.epoch = config['epoch']  #  10
        self.emb_dim = config['emb_dim']  #  10
        self.meta_dim = config['meta_dim']  #  50
        self.num_fields = config['num_fields']  #  2
        self.lr = config['lr']  #  0.001
        self.wd = config['wd']  #  0

        self.gcn_layer = 2  #  暂时设定层数为2
        self.temperature = config['temperature']  
        self.ssl_reg = config['ssl_reg']  

        record_path = './saved/' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + '/tgt_' + self.tgt + '_src_' + self.src + '_results/TransDICDR/'+'runid_'+str(self.runid)+'/'
        self.model_save_path = record_path + 'models/'
        self.pre_train_model_save_path = record_path + 'pre_train_models/'

        self.best_epoch = 0
        self.early_stop = 0
        self.results = {'Pretrain_s_mae': 10, 'Pretrain_s_rmse': 10, 'Pretrain_t_mae': 10, 'Pretrain_t_rmse': 10,'TransDICDR_mae': 10, 'TransDICDR_rmse': 10}
        
        
        ensureDir(record_path)
        ensureDir(self.model_save_path)
        ensureDir(self.pre_train_model_save_path)

        self.log = Logger(record_path)
        for key in config.keys():
            self.log.write(set_color(key + '=' + str(config[key]) + '\n', 'yellow'))

    def get_data(self):
        print('========Reading data========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)  #  <torch.utils.data.dataloader.DataLoader object at 0x7fa1c76e9f10>
        print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))  #  src 6631 iter / batchsize = 256 
        self.adj_matrix_s = self._lightgcn_adj_matrix(self.src_path)  #  <torch.utils.data.dataloader.DataLoader object at 0x7fa1c76e9f10>
        
        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))
        self.adj_matrix_t = self._lightgcn_adj_matrix(self.tgt_path)

        data_map = self.read_map_data() 
        print('map {} iter / batchsize = {} '.format(len(data_map), self.batchsize_map))  #  map 226 iter / batchsize = 64

        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))  #  test 734 iter / batchsize = 128

        return data_src, data_tgt, data_map, data_test

    def get_model(self):
        if self.base_model == 'LightGCN':
            model = TransDICDR(self.uid_all, self.iid_all, self.adj_matrix_s, self.adj_matrix_t, self.num_fields, self.emb_dim, self.gcn_layer, self.temperature, self.ssl_reg)  #  (181187, 114495, 2, 10, 50)
        else:
            raise ValueError('Unknown base model: ' + self.base_model)
        return model.cuda() if self.use_cuda else model

    def get_optimizer(self, model):
        optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd)  #  self.wd=0
        optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_map = torch.optim.Adam(params=model.mapping.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer_src, optimizer_tgt, optimizer_map

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
        for batch_idx, (X, y) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
            if mapping:  #  EMCDR
                ssl_loss, src_emb, tgt_emb = model(X, stage)
                #src_emb, tgt_emb = model(X, stage)
                mse_loss = criterion(src_emb, tgt_emb)
                loss = ssl_loss + mse_loss
                #loss = mse_loss
            else:
                pred = model(X, stage)
                loss = criterion(pred, y.squeeze().float())  #  与真实评分求分数差
            model.zero_grad()
            loss.backward()
            optimizer.step()

            current_batch = batch_idx + 1
            # if current_batch % 30 == 0:
            #     self.log.write(set_color('Train: Epoch:{:d}_current_batch:{:d}, loss:{:5f}, ssl_loss:{:.5f},  mse_loss:{:.5f}\n\n'
            #     .format(epoch, current_batch, loss, ssl_loss, mse_loss), 'red'))

    def update_results(self, model, epoch, mae, rmse, phase):
        if mae < self.results[phase + '_mae'] and rmse < self.results[phase + '_rmse']:
            self.early_stop = 0
            self.results[phase + '_mae'] = mae
            self.results[phase + '_rmse'] = rmse
            self.best_epoch = epoch
            best_ckpt = 'epoch_'+ str(epoch) + '_mae_' + str(self.results[phase + '_mae']) + '.ckpt'
            save_checkpoint(model, self.model_save_path, best_ckpt, max_checkpoints=3)

        elif mae < self.results[phase + '_mae']:
            self.early_stop = 0
            self.results[phase + '_mae'] = mae
            self.results[phase + '_rmse'] = rmse
            self.best_epoch = epoch
            best_ckpt = 'epoch_'+ str(epoch) + '_mae_' + str(self.results[phase + '_mae']) + '.ckpt'
            save_checkpoint(model, self.model_save_path, best_ckpt, max_checkpoints=3)

        else:
            self.early_stop += 1


    def CDR(self, model, data_src, data_tgt, data_map, data_test,
            criterion, optimizer_src, optimizer_tgt, optimizer_map):
        t1 = time()
        print('=====CDR Pretraining=====')
        print('Src Pretraining')
        for i in range(self.pre_train_epoch):
            self.train(data_src, model, criterion, optimizer_src, i, stage='train_src')
        print('Tgt Pretraining')
        for i in range(self.pre_train_epoch):
            self.train(data_tgt, model, criterion, optimizer_tgt, i, stage='train_tgt')
            if (i + 1) % 20 == 0: 
                save_checkpoint(model, self.pre_train_model_save_path, 'Epoch_i:' + str(i) + '.ckpt', max_checkpoints=5)
#         checkpoint = torch.load(self.pre_train_model_save_path + 'Epoch_i:39.ckpt')
#         model.load_state_dict(checkpoint)
        t2 = time()
        
        print('==========TransDICDR==========')
        for i in range(self.epoch):
            self.train(data_map, model, criterion, optimizer_map, i, stage='train_map', mapping=True)
            mae, rmse = self.eval_mae(model, data_test, stage='test_map')
            self.update_results(model, i, mae, rmse, 'TransDICDR')
            self.log.write(set_color('Eval: Epoch:{:d}, Test_tgt_MAE:{:.5f}, Test_tgt_RMSE:{:.5f}\n\n'
              .format(i, mae, rmse), 'blue'))
            if i >= 10  and self.early_stop >= 5:
                self.log.write(set_color('Early_stop_Epoch:{:d}\n'
              .format(i), 'blue'))
                break  
        t3 = time()
        self.log.write(set_color('Pre_train_Time:{:.5f}, Train_time:{:.5f}\n'
              .format((t2 - t1), (t3 - t2)), 'blue'))


    def main(self):
        data_src, data_tgt, data_map, data_test = self.get_data()  #  得到数据
        model = self.get_model()
        
        optimizer_src, optimizer_tgt, optimizer_map = self.get_optimizer(model)
        criterion = torch.nn.MSELoss()

        self.CDR(model, data_src, data_tgt, data_map, data_test,
                 criterion, optimizer_src, optimizer_tgt, optimizer_map)  #  跨域推荐
        
        self.log.write(set_color('TransDICDR: Best_Epoch:{:d}, Best_Tgt_MAE:{:.5f}, Best_Tgt_RMSE:{:.5f}\n'
              .format(self.best_epoch, self.results['TransDICDR_mae'], self.results['TransDICDR_rmse']), 'blue'))