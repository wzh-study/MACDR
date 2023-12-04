import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.Base_Module import *

# class ReverseLayerF(Function):

#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha

#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.alpha

#         return output, None


class EMCDR(torch.nn.Module):
    def __init__(self, data_config):
        super().__init__()
        self.num_fields = data_config['num_fields']  #  2
        self.uid_all = data_config['src_tgt_pairs'][data_config['task']]['uid']
        self.iid_all = data_config['src_tgt_pairs'][data_config['task']]['iid']
        self.share_user_num = data_config['src_tgt_pairs'][data_config['task']]['share_user_num']  #  3908
        
        self.latent_dim = data_config['latent_dim']  #  32

        self.n_mlplayer = data_config['n_mlplayer']

        #self.mapping = torch.nn.Linear(self.latent_dim, self.latent_dim, False)  #  不带bias  (10,10)
        layers = [nn.Linear(self.latent_dim, self.latent_dim, bias=False)] + [nn.Linear(self.latent_dim, self.latent_dim, bias=False) for _ in range(self.n_mlplayer - 1)]
        self.mapping = nn.Sequential(*layers)
        # self.DC = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.latent_dim // 2),  #  (200, 100)
        #     nn.Sigmoid(),
        #     nn.Linear(self.latent_dim // 2, 2),  #  二分类
        #     nn.Sigmoid()
        # )

        # self.cross_entropy = nn.CrossEntropyLoss()

    
    def InfoNCE_loss(self, view1, view2, temperature = 0.1, b_cos = True):
        if b_cos:
            view1 = F.normalize(view1, dim=1)
            view2 = F.normalize(view2, dim=1)
        
        # 计算正样本分数
        pos_score = torch.matmul(view1, view2.transpose(0, 1))  # (batch_size, all_user)
        pos_score = torch.exp(pos_score / temperature)

        # 计算全体样本的总分数
        ttl_score = torch.exp(torch.matmul(view1, view2.transpose(0, 1)) / temperature).sum(dim=1)  # (batch_size,)

        # 计算损失
        cl_loss = -torch.log(pos_score / (ttl_score.unsqueeze(1) + 1e-6))  # 加上小值避免除以零

        return torch.mean(cl_loss)
    
    
    def forward(self, x, stage):
        if stage == 'train_src':  #  pre-train
            emb = self.src_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        
        elif stage in ['train_tgt', 'test_tgt']:  #  only tgt
            emb = self.tgt_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)  #  torch.Size([256, 10]) * torch.Size([256, 10]) = torch.Size([256])
            return x
        # EMCDR
        elif stage == 'train_map':  #  EMCDR
            src_emb = self.src_model.user_embeddings(x.unsqueeze(1)).squeeze()  #  torch.Size([64, 10])
            src_emb_mapping = self.mapping.forward(src_emb)  #  torch.Size([64, 10])
            tgt_emb = self.tgt_model.user_embeddings(x.unsqueeze(1)).squeeze()  #  torch.Size([64, 10])  batch
            #tgt_emb = self.tgt_model.user_embeddings.weight

            #return src_emb_mapping, tgt_emb
            # DA loss
            # source_labels = torch.zeros(src_emb_mapping.shape[0]).long().cuda()  #  torch.Size([128])
            # target_labels = torch.ones(tgt_emb.shape[0]).long().cuda()  #  torch.Size([128])

            # source_reversed_feature = ReverseLayerF.apply(src_emb_mapping, 1.0)  #  batch_user_emb_t_mapping
            # source_class_output = self.DC(source_reversed_feature)

            # target_reversed_feature = ReverseLayerF.apply(tgt_emb, 1.0)   #  torch.Size([128, 32])  batch_user_emb_t
            # target_class_output = self.DC(target_reversed_feature)  #  torch.Size([128, 2])
            
            # return src_emb_mapping, tgt_emb, source_class_output, source_labels, target_class_output, target_labels
            return src_emb_mapping, tgt_emb

        elif stage == 'test_map':  #  EMCDR
            uid_emb = self.mapping.forward(self.src_model.user_embeddings(x[:, 0].unsqueeze(1)).squeeze())  #  torch.Size([128, 32])
            emb = self.tgt_model.forward(x)  #  torch.Size([128, 2, 32])
            emb[:, 0, :] = uid_emb  #  这里测试的embedding为源域对应映射过来的嵌入 物品为预训练过的目标域嵌入
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x



        # elif self.mode == "LightGCN":
        #     #  EMCDR LightGCN
        #     if stage == 'train_src':  #  pre-train  src
        #         user_emb_s, item_emb_s = self.src_model.forward()
        #         x = torch.sum(user_emb_s[x[:,0]] * item_emb_s[x[:,1]], dim=1)
        #         return x
            
        #     elif stage in ['train_tgt', 'test_tgt']:  #  pre-train  tgt
        #         user_emb_t, item_emb_t = self.tgt_model.forward()    
        #         x = torch.sum(user_emb_t[x[:,0]] * item_emb_t[x[:,1]], dim=1)  #  torch.Size([256, 10]) * torch.Size([256, 10]) = torch.Size([256])
        #         return x
        #     elif stage == 'train_map':  #  EMCDR
        #         user_emb_s, item_emb_s = self.src_model.forward()  #  torch.Size([123960, 32])——1.2137e+00,  4.2744e+00,  torch.Size([50053, 32])
        #         user_emb_t, item_emb_t = self.tgt_model.forward()
        #         batch_user_emb_s = user_emb_s[x].cuda()  #  torch.Size([64, 32])
        #         batch_user_emb_t = user_emb_t[x].cuda()

        #         src_emb = self.mapping.forward(batch_user_emb_s)  #  torch.Size([64, 32])
        #         tgt_emb = batch_user_emb_t  #  torch.Size([64, 32])   batch infonce
        #         #tgt_emb = user_emb_t  #  torch.Size([64, 32])    all infonce

        #         return src_emb, tgt_emb
            
        #     elif stage == 'test_map':  #  EMCDR
        #         user_emb_s, item_emb_s = self.src_model.forward()
        #         user_emb_t, item_emb_t = self.tgt_model.forward()
        #         uid_emb_s = self.mapping.forward(user_emb_s[x[:, 0]])  #  torch.Size([128, 32])
        #         x = torch.sum(uid_emb_s * item_emb_t[x[:, 1]], dim=1)
        #         return x
        
