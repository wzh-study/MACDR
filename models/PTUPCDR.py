import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.Base_Module import *

class MetaNet(torch.nn.Module):
    def __init__(self, latent_dim, meta_dim):
        super().__init__()
        #  Sequential((0): Linear(in_features=32, out_features=32, bias=True)  (1): ReLU()  (2): Linear(in_features=32, out_features=1, bias=False))
        self.event_K = torch.nn.Sequential(torch.nn.Linear(latent_dim, latent_dim), torch.nn.ReLU(), 
                                           torch.nn.Linear(latent_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(latent_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, latent_dim * latent_dim))  # Sequential((0): Linear(in_features=32, out_features=50, bias=True) (1): ReLU()  (2): Linear(in_features=50, out_features=320, bias=True))

    def forward(self, emb_fea, seq_index):   #  交互的正样本序列embedding:torch.Size([128, 20, 32])   交互的正样本序列:torch.Size([128, 20])
        mask = (seq_index == 0).float()  #  torch.Size([128, 20])  哪些是填充的为true
        event_K = self.event_K(emb_fea)  #  torch.Size([128, 20, 1])
        t = event_K - torch.unsqueeze(mask, 2) * 1e8  #  torch.Size([128, 20, 1])   true位置变成无限小  false位置不动
        att = self.event_softmax(t)  #  再softmax   torch.Size([128, 20, 1])
        his_fea = torch.sum(att * emb_fea, 1)  #  torch.Size([128, 32])
        output = self.decoder(his_fea)  #  torch.Size([128, 320])
        return output.squeeze(1)  #  torch.Size([128, 320])


class PTUPCDR(torch.nn.Module):
    def __init__(self, data_config):
        super().__init__()
        self.num_fields = data_config['num_fields']  #  2
        self.uid_all = data_config['src_tgt_pairs'][data_config['task']]['uid']
        self.iid_all = data_config['src_tgt_pairs'][data_config['task']]['iid']
        self.share_user_num = data_config['src_tgt_pairs'][data_config['task']]['share_user_num']  #  3908
        
        self.latent_dim = data_config['latent_dim']  #  32
        self.meta_dim = data_config['meta_dim']
        
        # PTUPCDR
        self.meta_net = MetaNet(self.latent_dim, self.meta_dim)  #  (32, 50)
        
    
    def forward(self, x, stage):
        #  PTUPCDR
        if stage in ['train_meta', 'test_meta']:  #  PTUPCDR  x:torch.Size([128, 22])
            iid_emb = self.tgt_model.item_embeddings(x[:, 1].unsqueeze(1))  #  torch.Size([128, 1, 32])     
            uid_emb_src = self.src_model.user_embeddings(x[:, 0].unsqueeze(1))  #  torch.Size([128, 1, 32])
            ufea = self.src_model.item_embeddings(x[:, 2:])  #  torch.Size([128, 20, 32])    
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.latent_dim, self.latent_dim)  #  torch.Size([128, 32, 32])  attention学习pu 并且得到学习参数wu
            uid_emb = torch.bmm(uid_emb_src, mapping)  #  torch.Size([128, 1, 32])
            emb = torch.cat([uid_emb, iid_emb], 1)  #  torch.Size([128, 2, 32])
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        

        # elif self.mode == "LightGCN":
        #     if stage == 'train_src':  #  pre-train  src
        #         user_emb_s, item_emb_s = self.src_model.forward()
        #         x = torch.sum(user_emb_s[x[:,0]] * item_emb_s[x[:,1]], dim=1)
        #         return x
            
        #     elif stage in ['train_tgt', 'test_tgt']:  #  pre-train  tgt
        #         user_emb_t, item_emb_t = self.tgt_model.forward()
        #         x = torch.sum(user_emb_t[x[:,0]] * item_emb_t[x[:,1]], dim=1)  #  torch.Size([256, 10]) * torch.Size([256, 10]) = torch.Size([256])
        #         return x
            

        #     elif stage in ['train_meta', 'test_meta']:   #  x: torch.Size([128, 22])
        #         user_emb_s, item_emb_s = self.src_model.forward()
        #         user_emb_t, item_emb_t = self.tgt_model.forward()

        #         #  pos item
        #         iid_emb = item_emb_t[x[:, 1].unsqueeze(1)]  #  torch.Size([bs, 1, 32])   bs=128     
            
        #         #  pretrain Pu 
        #         user_src_emb = user_emb_s[x[:, 0]].unsqueeze(1)  #  torch.Size([bs, 1, 32])

        #         #  source interaction seq
        #         ufea = item_emb_s[x[:, 2:]]  #  torch.Size([bs, 20, 32])

        #         mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.latent_dim, self.latent_dim)  #  torch.Size([128, 32, 32])  attention学习pu 并且得到学习参数wu
        #         uid_emb = torch.bmm(user_src_emb, mapping)  #  torch.Size([128, 1, 32])

        #         emb = torch.cat([uid_emb, iid_emb], 1)  #  torch.Size([128, 2, 32])
        #         output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
                
        #         return output

            