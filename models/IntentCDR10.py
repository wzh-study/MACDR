import torch
import random
import faiss
import torch.nn.functional as F
import torch.nn as nn
from models.Base_Module import *


class IntentCDR10(torch.nn.Module):
    def __init__(self, data_config, user_interactions):
        super().__init__()
        self.num_fields = data_config['num_fields']  #  2
        self.uid_all = data_config['src_tgt_pairs'][data_config['task']]['uid']
        self.iid_all = data_config['src_tgt_pairs'][data_config['task']]['iid']
        self.share_user_num = data_config['src_tgt_pairs'][data_config['task']]['share_user_num']  #  3908
        
        self.latent_dim = data_config['latent_dim']  #  32
        self.meta_dim = data_config['meta_dim']
        self.temperature = data_config['temperature']

        self.num_cluster = data_config['num_cluster']
        self.closest_center = data_config['closest_center']

        self.margin = data_config['margin']
        self.lamda = data_config['lamda']
        self.neg_sample = data_config['neg_sample']
        self.dropout = data_config['dropout']

        #  所有源域用户的交互序列
        self.user_interactions = user_interactions
        #  最大交互序列的长度
        self.max_sequence_length = 20

        
        #  多任务
        self.n_task = data_config['n_task']  #  任务数量
        self.n_expert = data_config['n_expert']  #  底层专家数量
        self.expert_params = {"dims": [self.latent_dim, self.latent_dim, self.latent_dim, self.latent_dim]}  
        self.n_mlplayer = data_config['n_mlplayer']
        

        #  attention src item seq
        self.event_K = torch.nn.Sequential(torch.nn.Linear(self.latent_dim, self.latent_dim), torch.nn.ReLU(), 
                                           torch.nn.Linear(self.latent_dim, 1, False))
        
        # #  Experts
        self.experts = nn.ModuleList(
            MLP(2 * self.latent_dim, output_layer=False, **self.expert_params) for i in range(self.n_expert))
        
        # # # #  Gates
        self.gate = nn.ModuleList(MLP(self.latent_dim, output_layer=False, **{
                "dims": [self.n_expert],
                "activation": "softmax"}) for i in range(self.n_expert)) 
        

        # self.mapping = MLPLayers(layers=[2 * self.latent_dim, self.latent_dim, self.latent_dim],
        #                                     activation='None', dropout=0, bn=False)
        #  四种方式   相似度内积margin_loss / GAN_loss / KL散度
        self.discriminator = nn.Sequential(
            nn.Linear(2 * self.latent_dim, self.latent_dim),
            nn.ReLU(),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.ReLU(),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def run_kmeans(self, x, latent_dim, num_cluster):
        '''
        Run K-means algorithm to get k clusters of the input x
        Faiss 库实现了 K-means 聚类算法。
        具体来说，输入是一个数据矩阵 x,每行代表一个数据样本,num_cluster 表示期望聚成的簇数。
        然后使用 faiss.Kmeans 类，指定聚类的维数 latent_dim、簇数 num_cluster 和 gpu=True,表示使用 GPU 进行计算。
        接着,用 kmeans.train(x) 方法训练 K-means 模型，得到每个簇的中心向量 cluster_cents。
        最后,使用 kmeans.index.search(x, 1) 方法，得到每个数据样本所属的簇的索引，即 I 数组。
        '''
        kmeans = faiss.Kmeans(d=latent_dim, k=num_cluster)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, self.closest_center)  #  每个用户或者物品最近的self.closest_center个聚类中心下标
        
        #  convert to cuda Tensor for broadcasrt
        centroids = torch.Tensor(cluster_cents).cuda()
        centroids = F.normalize(centroids, p=2, dim=1)  #  归一化聚类中心坐标
        node2cluster = torch.LongTensor(I).squeeze().cuda() 

        return centroids, node2cluster
    
    def e_s_step(self):
        user_embeddings = self.src_model.user_embeddings.weight.detach().cpu().numpy()
        
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings, self.latent_dim, self.num_cluster)  #  torch.Size([200, 32])  torch.Size([73277, 5])
        # 提取每个用户相关的前topk个簇中心嵌入
        top_k_user_centroids = self.user_centroids[self.user_2cluster]  #  torch.Size([73277, k, 32])   if clostest=1, torch.Size([35598, 32])
        
        
        # if self.closest_center == 1:
        #     top_k_user_centroids = top_k_user_centroids.unsqueeze(1)

        top_k_user_centroids = F.normalize(top_k_user_centroids, p=2, dim=1).cuda()  #  (all_user, k, latent_dim) 
        #  非重叠用户  每个聚类中心对应的类
        self.centroid_to_users = {}

        # for i in range(self.share_user_num, top_k_user_centroids.shape[0]):
        #     user_centroid = top_k_user_centroids[i].squeeze(dim=0).tolist()

        #     # 将用户添加到对应聚类中心的列表中
        #     if tuple(user_centroid) not in self.centroid_to_users:
        #         self.centroid_to_users[tuple(user_centroid)] = []

        #     self.centroid_to_users[tuple(user_centroid)].append(i)

        for i in range(self.share_user_num, top_k_user_centroids.shape[0]):
            user_centroid_sum = top_k_user_centroids[i].sum().item()

            # 将用户添加到对应聚类中心的列表中
            if user_centroid_sum not in self.centroid_to_users:
                self.centroid_to_users[user_centroid_sum] = []

            self.centroid_to_users[user_centroid_sum].append(i)

        self.user_topk_centroids = top_k_user_centroids.cuda()
        
        


    def sample_users_with_same_and_diff_class(self, batch_user_ids, batch_size):
        # 获取用户总数和表征维度
        num_users, latent_dim = self.user_topk_centroids.size()

        # 初始化结果列表
        same_class_users = []
        diff_class_users = []

        for user_idx in range(len(batch_user_ids)):
            user_id = batch_user_ids[user_idx].item()

            # 获取当前用户所属的聚类中心
            user_centroid_tensor = self.user_topk_centroids[user_id].squeeze(dim=0)     # torch.Size([32])
            user_centroid_list = user_centroid_tensor.sum().item()
            # 从相同类用户中随机选择一个非重叠用户
            if user_centroid_list in self.centroid_to_users:
                same_class_candidates = self.centroid_to_users[user_centroid_list]
                if same_class_candidates:
                    same_class_user = random.choice(same_class_candidates)
                    while same_class_user == user_id:  #  等于当前值就要重新抽
                        same_class_user = random.choice(same_class_candidates)
                    same_class_users.append(same_class_user)
            else:
                same_class_users.append(user_id)

            # 从不同类用户中随机选择一个，确保其聚类中心与当前用户不一致
            neg_samples = []

            while len(neg_samples) < self.neg_sample:
                # 随机选择一个用户
                diff_class_user = random.choice(range(num_users))
                diff_class_user_centroid = self.user_topk_centroids[diff_class_user].squeeze(dim=0)
                # 如果采样到的用户不是当前用户并且与当前用户的聚类中心不相同
                if diff_class_user > self.share_user_num and not torch.equal(user_centroid_tensor, diff_class_user_centroid):
                    neg_samples.append(diff_class_user)
            
            
            # 将负样本列表转换为张量
            neg_samples_tensor = torch.tensor(neg_samples, dtype=torch.long)
    
            # 存储当前用户的负样本张量到列表中
            diff_class_users.append(neg_samples_tensor)

        # 将负样本列表堆叠成一个张量
        same_class_users = torch.tensor(same_class_users, dtype=torch.long).cuda()
        diff_class_users = torch.stack(diff_class_users, dim=0).cuda()  #  torch.Size([256, 1])
            
        # 截取结果列表，确保返回的用户数量为 batch_size
        same_class_users = same_class_users[:batch_size]
        diff_class_users = diff_class_users[:batch_size]

        return same_class_users, diff_class_users  #  torch.Size([256])  torch.Size([256, 1])


    def extract_and_concat_user_interactions(self, user_interactions, user_ids):
        user_interactions_tensors = []

        user_ids_ = user_ids
        if user_ids.dim() == 2:
            # 如果 user_ids 是二维张量 (bs, neg_sample)
            user_ids = user_ids.view(-1)  # 将其展平成一维张量

        for user_id in user_ids:
            interactions = user_interactions.get(user_id.item(), [])  # 获取用户的交互物品列表

            # 将交互物品列表转换为张量
            interactions_tensor = torch.tensor(interactions, dtype=torch.long)

            # 截取或填充序列，使其具有相同的长度
            if len(interactions_tensor) < self.max_sequence_length:
                pad_length = self.max_sequence_length - len(interactions_tensor)
                interactions_tensor = torch.cat((interactions_tensor, torch.zeros(pad_length, dtype=torch.long)))

            interactions_tensor = interactions_tensor[:self.max_sequence_length]

            user_interactions_tensors.append(interactions_tensor)

        # 将用户交互张量连接成一个张量
        concatenated_interactions = torch.stack(user_interactions_tensors)  # torch.Size([bs, 20])   torch.Size([ns*neg_sample, 20])

        # 如果输入是 (bs, neg_sample)，则展开成 (bs, neg_sample, seq_len)
        # if user_ids_.dim() == 2:
        #     concatenated_interactions = concatenated_interactions.view(user_ids_.shape[0], self.neg_sample, 20)

        return concatenated_interactions.cuda()



    def attention(self, seq_index):
        '''
        seq_index: 交互的序列中物品ID对应列表  torch.size([bs, seq_len(20)])
        '''
        #  source interaction seq
        ufea = self.src_model.item_embeddings(seq_index)  #  torch.Size([128, 20, 32])  
        #  mask   
        mask = (seq_index == 0).float()  #  torch.Size([256, 20])  哪些是填充的为true
        #  attention
        event_K = self.event_K(ufea)  #  torch.Size([256, 20, 1])
        t = event_K - torch.unsqueeze(mask, 2) * 1e8  #  torch.Size([256, 20, 1])   true位置变成无限小  false位置不动
        att = F.softmax(t, dim=1)  #  再softmax   torch.Size([256, 20, 1])
        his_fea = torch.sum(att * ufea, 1)  #  torch.Size([256, 32])  pu

        return his_fea


    def MOE(self, expert_input, gate_input):
        #  提取共享信息
        expert_outs = [expert(expert_input).unsqueeze(1) for expert in self.experts]
        expert_outs = torch.cat(expert_outs, dim=1)  #  torch.Size([256, 2, 32])   
        
        gate_outs_share = [gate(gate_input).unsqueeze(-1) for gate in self.gate]  #  1 x torch.Size([256, 2, 1])
        for gate_out in gate_outs_share:
            expert_weight = torch.mul(gate_out, expert_outs)  #  torch.Size([256, 2, 32])
            expert_pooling = torch.sum(expert_weight, dim=1)  #  torch.Size([256, 32])
        
        return expert_pooling

    # def compute_margin_loss(self, margin, user_emb, pos_emb, neg_emb):
    #     pos_scores = torch.sum(user_emb * pos_emb, dim=1)
    #     neg_scores = torch.sum(user_emb * neg_emb, dim=1)

    #     margin_loss = torch.mean(torch.max(torch.tensor(0.0), torch.tensor(margin) - (pos_scores - neg_scores)))
        
    #     return margin_loss
    def compute_margin_loss(self, user_emb, pos_emb, neg_emb, margin):  #  torch.Size([512, 32])  torch.Size([512, 32])
        # 计算正样本的分数
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        # 计算负样本的分数
        neg_scores = torch.sum(user_emb.unsqueeze(1) * neg_emb.view(-1, self.neg_sample, self.latent_dim), dim=2)
        # 计算每个负样本的损失
        per_sample_loss = torch.max(torch.tensor(0.0), torch.tensor(margin) - (pos_scores.view(-1, 1) - neg_scores))
        # 将所有样本的损失取平均
        margin_loss = torch.mean(per_sample_loss)

        return margin_loss

    def compute_kl_loss(self, user_emb, pos_emb, neg_emb):
        # 计算正样本的分数
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)

        # 计算负样本的分数
        neg_scores = torch.sum(user_emb.unsqueeze(1) * neg_emb.view(-1, self.neg_sample, self.latent_dim), dim=2)

        # 计算 KL 散度项
        kl_loss = F.kl_div(F.log_softmax(pos_scores, dim=0), F.softmax(neg_scores, dim=0))

        return kl_loss
    
    
    @staticmethod
    def embedding_normalize(embeddings):
        emb_length = torch.sum(embeddings**2, dim=1, keepdim=True)
        ones = torch.ones_like(emb_length)
        norm = torch.where(emb_length > 1, emb_length, ones)
        return embeddings / norm
    
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
        
        elif stage == 'train_meta':  #  PTUPCDR  x:torch.Size([128, 22])
            #  target-domain batch item embedding
            iid_emb = self.tgt_model.item_embeddings(x[:, 1])  #  torch.Size([128, 32])     
            #  pretrain batch Pu 
            user_src_emb = self.src_model.user_embeddings(x[:, 0])  #  torch.Size([bs, 32])
            

            #  attention Pu
            overlap_attention_item_seq_emb = self.attention(x[:, 2:])
        

            #  user src intent embedding
            batch_user_ids = x[:, 0]
            user_src_intent_emb = self.user_topk_centroids[batch_user_ids]  #  torch.Size([256, 1, 32])

            #  采样  聚类相同的非重叠源域用户 与 聚类不同的非重叠源域用户   (batch_size,)  torch.Size([256])   torch.Size([256, neg_sample])
            same_class_users, diff_class_users = self.sample_users_with_same_and_diff_class(batch_user_ids, batch_user_ids.shape[0])
            
            #  提取相同类用户和不同类用户的交互列表  #  torch.Size([256, 20])  torch.Size([256 * neg_sample, 20])
            same_class_interactions = self.extract_and_concat_user_interactions(self.user_interactions, same_class_users)
            diff_class_interactions = self.extract_and_concat_user_interactions(self.user_interactions, diff_class_users)

            #  same and diss class src_user_emb  /  attention item seq emb  /  intent_emb
            same_class_user_src_emb = self.src_model.user_embeddings(same_class_users)  #  torch.Size([256, 32])
            same_class_attention_item_seq_emb = self.attention(same_class_interactions)  #  torch.Size([256, 32])
            same_class_user_src_intent_emb = user_src_intent_emb  #  torch.Size([256, 32])

            diff_class_user_src_emb = self.src_model.user_embeddings(diff_class_users).view(-1, 32)  #  torch.Size([256 * 5, 32])
            diff_class_attention_item_seq_emb = self.attention(diff_class_interactions)  #   torch.Size([1280, 32])
            diff_class_user_src_intent_emb = self.user_topk_centroids[diff_class_users].view(-1, 32)  #  torch.Size([256 * 5, 32])


            #  overlap数据: gate和expert的输入
            overlap_concatenated_tensor = torch.cat((user_src_emb, overlap_attention_item_seq_emb), dim=1)  #  torch.Size([256, 2 * 32])
            #overlap_mean_tensor = torch.mean(torch.stack([user_src_emb, overlap_attention_item_seq_emb], dim=0), dim=0)
            
            overlap_expert_tensor_input = overlap_concatenated_tensor
            overlap_gate_tensor_input = user_src_intent_emb  #  gate1控制的是冷启动的任务，需要让相似的用户拥有相似的gate权重 是为了更好的映射函数设置
            #overlap_gate_tensor_input = overlap_concatenated_tensor

            overlap_expert_src_mapping_output = self.MOE(overlap_expert_tensor_input, overlap_gate_tensor_input)  #  torch.Size([256, 32])

            #  采样的gate和expert
            same_class_concatenated_tensor = torch.cat((same_class_user_src_emb, same_class_attention_item_seq_emb), dim=1)
            same_class_expert_tensor_input = same_class_concatenated_tensor
            same_class_gate_tensor_input = same_class_user_src_intent_emb
            #same_class_gate_tensor_input = same_class_concatenated_tensor
            same_class_expert_src_mapping_output = self.MOE(same_class_expert_tensor_input, same_class_gate_tensor_input)


            diff_class_concatenated_tensor = torch.cat((diff_class_user_src_emb, diff_class_attention_item_seq_emb), dim=1)
            diff_class_expert_tensor_input = diff_class_concatenated_tensor
            diff_class_gate_tensor_input = diff_class_user_src_intent_emb
            #diff_class_gate_tensor_input = diff_class_concatenated_tensor
            diff_class_expert_src_mapping_output = self.MOE(diff_class_expert_tensor_input, diff_class_gate_tensor_input)  #  torch.Size([1280, 32])

            # loss_a = self.compute_kl_loss(self.embedding_normalize(overlap_expert_src_mapping_output),
            #                     self.embedding_normalize(same_class_expert_src_mapping_output),
            #                     self.embedding_normalize(diff_class_expert_src_mapping_output))
            #loss_a = self.compute_margin_loss(overlap_expert_src_mapping_output,same_class_expert_src_mapping_output, diff_class_expert_src_mapping_output, self.margin)
            
            # return overlap_expert_src_mapping_output, iid_emb
            # return overlap_expert_src_mapping_output, iid_emb
            return overlap_expert_src_mapping_output, iid_emb, same_class_expert_src_mapping_output, diff_class_expert_src_mapping_output
        
        elif stage == "test_meta":
            #  target-domain batch item embedding
            iid_emb = self.tgt_model.item_embeddings(x[:, 1].unsqueeze(1))  #  torch.Size([128, 1, 32])     
            #  pretrain batch Pu 
            user_src_emb = self.src_model.user_embeddings(x[:, 0])  #  torch.Size([bs, 32])
            
            #  attention Pu
            overlap_attention_item_seq_emb = self.attention(x[:, 2:])
            
            batch_user_ids = x[:, 0]

            #  user src intent embedding
            user_src_intent_emb = self.user_topk_centroids[batch_user_ids]  # 形状为 (bs, 32)

            #  overlap数据: gate和expert的输入
            overlap_concatenated_tensor = torch.cat((user_src_emb, overlap_attention_item_seq_emb), dim=1)  #  torch.Size([256, 2 * 32])
            #overlap_mean_tensor = torch.mean(torch.stack([user_src_emb, overlap_attention_item_seq_emb], dim=0), dim=0)
            
            overlap_expert_tensor_input = overlap_concatenated_tensor
            overlap_gate_tensor_input = user_src_intent_emb  #  gate1控制的是冷启动的任务，需要让相似的用户拥有相似的gate权重 是为了更好的映射函数设置
            #overlap_gate_tensor_input = overlap_concatenated_tensor

            overlap_expert_src_mapping_output = self.MOE(overlap_expert_tensor_input, overlap_gate_tensor_input)  #  torch.Size([256, 32])

            uid_emb = overlap_expert_src_mapping_output.unsqueeze(1)  #  torch.Size([256, 1, 32])

                
            emb = torch.cat([uid_emb, iid_emb], 1)  #  torch.Size([128, 2, 32])
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)

            return output

        