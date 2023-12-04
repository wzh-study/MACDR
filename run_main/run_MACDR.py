import torch
import tqdm
from time import time
from data_process.Load_data import Datasets
from itertools import zip_longest
from models.MACDR import *
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
        self.num_fields = config['num_fields']  #  2
        
        self.latent_dim = config['latent_dim']
        self.lr = config['lr']  #  0.001
        self.lr_meta = config['lr_meta']
        self.weight_decay = config['weight_decay']  #  0
        self.config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.temperature = config['temperature']
        self.lamda = config['lamda']
        self.l2_reg = config['l2_reg']
        self.source_domain_reg  = config['source_domain_reg']

        record_path = './saved/' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + '/tgt_' + self.tgt + '_src_' + self.src + '_results/MACDR/'+'runid_'+str(self.runid)+'/'
        self.model_save_path = record_path + 'models/'
        self.model_save_pre_train_path = record_path + 'pretrain_models/'

        self.best_epoch = 0
        self.early_stop = 0
        self.results = {'tgt_mae': 10, 'tgt_rmse': 10,
                        'MACDR_mae': 10, 'MACDR_rmse': 10}
        
        ensureDir(record_path)
        ensureDir(self.model_save_path)
        ensureDir(self.model_save_pre_train_path)

        self.log = Logger(record_path)
        for key in config.keys():
            self.log.write(set_color(key + '=' + str(config[key]) + '\n', 'yellow'))

    def get_data(self):
        print('========Reading data========')
        data_src, user_interactions = self.read_log_data_seqsrc(self.src_path, self.batchsize_src)  #  <torch.utils.data.dataloader.DataLoader object at 0x7fa1c76e9f10>
        print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))  #  src 6631 iter / batchsize = 256 
        #self.adj_matrix_s = self._lightgcn_adj_matrix(self.src_path)  #  <torch.utils.data.dataloader.DataLoader object at 0x7fa1c76e9f10>
        
        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))
        #self.adj_matrix_t = self._lightgcn_adj_matrix(self.tgt_path)

        #  PUTPCDR
        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta, history=True)
        print('meta {} iter / batchsize = {} '.format(len(data_meta), self.batchsize_meta))  #  meta 2924 iter / batchsize = 128

        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))  #  test 734 iter / batchsize = 128

        return data_src, user_interactions, data_tgt, data_meta, data_test

    def get_model(self, user_interactions):
        if self.base_model == 'MF':
            pre_train_model = Pre_Train(self.config, self.uid_s, self.iid_s, self.uid_t, self.iid_t)
        
        else:
            raise ValueError('Unknown base model: ' + self.base_model)
        
        model = MACDR(self.config, user_interactions)  #  (181187, 114495, 2, 10, 50)
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
            if mapping:  
                src_emb, tgt_emb = model(X, stage)
                loss = criterion(src_emb, tgt_emb)

            else:  #  MACDR
                pred = model(X, stage)
                #pred, pred_pos, pred_neg = model(X, stage)
                loss = criterion(pred, y.squeeze().float())   #  与真实评分求分数差

            model.zero_grad()
            loss.backward()
            optimizer.step()

    def train_MACDR(self, data_loader1, data_loader2, model, criterion, optimizer_g, GAN_loss, optimizer_d, epoch, stage, mapping=False):
        print('Training Epoch {}:'.format(epoch))
        all_d_loss, all_g_loss, all_main_loss, all_source_loss, all_loss, batch_num = 0, 0, 0, 0, 0, 0
        model.train()

        if stage == 'train_meta' and epoch == 0:
            model.e_s_step()

        # source_loss = 0
        # for (X1, y1) in tqdm.tqdm(data_loader1, smoothing=0, mininterval=1.0):
        #     pred_s = model(X1, 'train_src')
        #     source_loss += criterion(pred_s, y1.squeeze().float())

        # source_loss = self.source_domain_reg * source_loss_part
        
        #for (X1, y1), (X2, y2) in tqdm.tqdm(zip(data_loader1, data_loader2), smoothing=0, mininterval=1.0):
        for (X2, y2) in tqdm.tqdm(data_loader2, smoothing=0, mininterval=1.0):
            ##  Train Discriminator
            #overlap_expert_src_mapping_output, iid_mapping_emb, loss_a = model(X, stage)

            
            overlap_expert_src_mapping_output, iid_mapping_emb, same_class_expert_src_mapping_output, diff_class_expert_src_mapping_output = model(X2, stage)
            overlap_expert_src_mapping_output, iid_mapping_emb, same_class_expert_src_mapping_output, diff_class_expert_src_mapping_output = overlap_expert_src_mapping_output.detach(), iid_mapping_emb.detach(), same_class_expert_src_mapping_output.detach(), diff_class_expert_src_mapping_output.detach()
            # 先更新判别器损失  不加detach会报大错！！！
            optimizer_d.zero_grad()
            pos_same_class_concat_output = torch.cat((overlap_expert_src_mapping_output, same_class_expert_src_mapping_output), dim=1)
            neg_diff_class_concat_output = torch.cat((overlap_expert_src_mapping_output.repeat(1, self.config['neg_sample']).view(-1, self.latent_dim), diff_class_expert_src_mapping_output), dim=1)
            real_labels = torch.ones(len(pos_same_class_concat_output), 1).cuda()
            fake_labels = torch.zeros(len(neg_diff_class_concat_output), 1).cuda()
            real_loss = GAN_loss(model.discriminator(pos_same_class_concat_output), real_labels)
            fake_loss = GAN_loss(model.discriminator(neg_diff_class_concat_output), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()
        
        
            # if epoch >= 3 and epoch % 2 == 0:
            optimizer_g.zero_grad()
            overlap_expert_src_mapping_output, iid_mapping_emb, same_class_expert_src_mapping_output, diff_class_expert_src_mapping_output = model(X2, stage)
            pred = torch.sum(overlap_expert_src_mapping_output * iid_mapping_emb, dim=1)
            #  main loss
            g_loss_main = criterion(pred, y2.squeeze().float())    #  与真实评分求分数差     
            pos_same_class_concat_output = torch.cat((overlap_expert_src_mapping_output, same_class_expert_src_mapping_output), dim=1)
            neg_diff_class_concat_output = torch.cat((overlap_expert_src_mapping_output.repeat(1, self.config['neg_sample']).view(-1, self.latent_dim), diff_class_expert_src_mapping_output), dim=1)
            real_labels = torch.ones(len(pos_same_class_concat_output), 1).cuda()
            fake_labels = torch.zeros(len(neg_diff_class_concat_output), 1).cuda()
            real_loss = GAN_loss(model.discriminator(pos_same_class_concat_output), real_labels)
            fake_loss = GAN_loss(model.discriminator(neg_diff_class_concat_output), fake_labels)
            g_loss_d = real_loss + fake_loss
            
        
            #  source_domain loss
            # pred_s = model(X1, 'train_src')
            # source_loss = criterion(pred_s, y1.squeeze().float())
            source_loss = torch.tensor(0., requires_grad=True)
            #  reg_loss
            reg_loss = torch.tensor(0., requires_grad=True)
            for param in model.parameters():
                reg_loss = reg_loss + torch.norm(param, 2)

            g_loss = g_loss_main + self.l2_reg * reg_loss + self.source_domain_reg * source_loss - self.lamda * g_loss_d 
            g_loss.backward()
            optimizer_g.step()


            # optimizer_d.zero_grad()
            # real_labels = torch.ones(len(overlap_expert_src_mapping_output), 1).cuda()
            # fake_labels = torch.zeros(len(diff_class_expert_src_mapping_output), 1).cuda()
            # real_loss_1 = GAN_loss(model.discriminator(overlap_expert_src_mapping_output), real_labels)
            # real_loss_2 = GAN_loss(model.discriminator(same_class_expert_src_mapping_output), real_labels)
            # fake_loss = GAN_loss(model.discriminator(diff_class_expert_src_mapping_output), fake_labels)
            # d_loss = real_loss_1 + real_loss_2 + fake_loss
            # d_loss.backward()
            # optimizer_d.step()


            # optimizer_g.zero_grad()
            # overlap_expert_src_mapping_output, iid_mapping_emb, same_class_expert_src_mapping_output, diff_class_expert_src_mapping_output = model(X, stage)
            # pred = torch.sum(overlap_expert_src_mapping_output * iid_mapping_emb, dim=1)
            # g_loss_main = criterion(pred, y.squeeze().float())    #  与真实评分求分数差     
            # real_loss_1 = GAN_loss(model.discriminator(overlap_expert_src_mapping_output), real_labels)
            # real_loss_2 = GAN_loss(model.discriminator(same_class_expert_src_mapping_output), real_labels)
            # fake_loss = GAN_loss(model.discriminator(diff_class_expert_src_mapping_output), fake_labels)
            # g_loss_d = real_loss_1 + real_loss_2 + fake_loss
            # g_loss = g_loss_main - self.lamda * g_loss_d 
            # g_loss.backward()
            # optimizer_g.step()

            # if epoch > 0 and epoch % 3 == 0:
            #     ##  Train Generator  更新生成器损失
            #     optimizer_g.zero_grad()
            #     overlap_expert_src_mapping_output, iid_mapping_emb, same_class_expert_src_mapping_output, diff_class_expert_src_mapping_output = model(X, stage)
            #     pred = torch.sum(overlap_expert_src_mapping_output * iid_mapping_emb, dim=1)
            #     g_loss_main = criterion(pred, y.squeeze().float())    #  与真实评分求分数差
            #     pos_same_class_concat_output = torch.cat((overlap_expert_src_mapping_output, same_class_expert_src_mapping_output), dim=1)
            #     neg_diff_class_concat_output = torch.cat((overlap_expert_src_mapping_output.repeat(1, self.config['neg_sample']).view(-1, self.latent_dim), diff_class_expert_src_mapping_output), dim=1)
            #     real_labels = torch.ones(len(pos_same_class_concat_output), 1).cuda()
            #     fake_labels = torch.zeros(len(neg_diff_class_concat_output), 1).cuda()
            #     real_loss = GAN_loss(model.discriminator(pos_same_class_concat_output), real_labels)
            #     fake_loss = GAN_loss(model.discriminator(neg_diff_class_concat_output), fake_labels)
            #     g_loss_d = real_loss + fake_loss
            #     g_loss = g_loss_main + self.lamda * g_loss_d 
            #     # l2_reg = torch.tensor(0., requires_grad=True)
            #     # for param in model.parameters():
            #     #     l2_reg = l2_reg + torch.norm(param, 2)
            #     # loss = g_loss + self.lamda * d_loss + self.l2_reg * l2_reg
            #     g_loss.backward()
            #     optimizer_g.step()
            
            # else:
            #     g_loss = torch.tensor(0)

            loss = d_loss + g_loss

            all_d_loss += d_loss.item()
            all_g_loss += g_loss.item()
            all_main_loss += g_loss_main.item()
            all_source_loss += source_loss.item()
            all_loss += loss.item()
            
            batch_num += 1
            
        self.log.write(set_color('Epoch:{:d}, Loss_d:{:.4f}, Loss_g:{:.4f}, Loss_main:{:.4f} , Loss_source:{:.4f}, Loss_all:{:.4f}\n'
              .format(epoch, all_d_loss / batch_num, all_g_loss / batch_num, all_main_loss / batch_num, all_source_loss / batch_num, all_loss / batch_num), 'red'))     


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

        #  预训练赋值
        model.src_model = pre_train_model.src_model
        model.tgt_model = pre_train_model.tgt_model

        #  源域参数不更新为False
        for param in model.src_model.parameters():
            param.requires_grad = False
        # for param in model.src_model.item_embeddings.parameters():
        #     param.requires_grad = False
        
        for param in model.discriminator.parameters():
            param.requires_grad = False

        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
        optimizer_meta = torch.optim.Adam(params=params_to_optimize, lr=self.lr_meta, weight_decay=self.weight_decay)
        
        for param in model.discriminator.parameters():
            param.requires_grad = True
            
        GAN_loss = nn.BCEWithLogitsLoss()
        optimizer_d = torch.optim.Adam(params=model.discriminator.parameters(), lr=self.lr_meta, weight_decay=self.weight_decay)


        print('\n model parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        t2 = time()
        print('==========MACDR==========')
        for i in range(self.epoch):
            self.train_MACDR(data_src, data_meta, model, criterion, optimizer_meta, GAN_loss, optimizer_d, i, stage='train_meta')
            mae, rmse = self.eval_mae(model, data_test, stage='test_meta')

            self.update_results(model, i, mae, rmse, 'MACDR', self.model_save_path)
            self.log.write(set_color('Eval: Epoch:{:d}, Test_tgt_MAE:{:.5f}, Test_tgt_RMSE:{:.5f}\n\n'
              .format(i, mae, rmse), 'blue'))
            if i >= 10  and self.early_stop >= 10:
                self.log.write(set_color('Early_stop_Epoch:{:d}\n'
              .format(i), 'blue'))
                break  
        t3 = time()
        self.log.write(set_color('Pre_train_Time:{:.5f}, Train_time:{:.5f}\n'
              .format((t2 - t1), (t3 - t2)), 'blue'))

   

    def main(self):
        data_src, user_interactions, data_tgt, data_meta, data_test = self.get_data()  #  得到数据
        pre_train_model, model = self.get_model(user_interactions)
        optimizer_src, optimizer_tgt = self.get_optimizer(pre_train_model)  #  预训练优化器
        criterion = torch.nn.MSELoss()
        
        
        self.CDR(pre_train_model, model, data_src, data_tgt, data_meta, data_test,
                 criterion, optimizer_src, optimizer_tgt)  #  跨域推荐
        
        self.log.write(set_color('MACDR: Best_Epoch:{:d}, Best_Tgt_MAE:{:.5f}, Best_Tgt_RMSE:{:.5f}\n'
              .format(self.best_epoch, self.results['MACDR_mae'], self.results['MACDR_rmse']), 'blue'))