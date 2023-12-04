import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import pickle
import numpy as np
import scipy.sparse as sp
from tensorflow import keras

class Datasets():
    def __init__(self,
                 config):
        self.use_cuda = config['use_cuda']  #  0

        self.base_model = config['base_model']  #  'MF'
        self.root = config['root']  #  './data/'
        self.ratio = config['ratio']  #  [0.8, 0.2]
        self.task = config['task']  #  '1'
        self.src = config['src_tgt_pairs'][self.task]['src']  #  'Movies_and_TV'
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']  #  'CDs_and_Vinyl'


        self.uid_s = config['src_tgt_pairs'][self.task]['uid_s']  
        self.iid_s = config['src_tgt_pairs'][self.task]['iid_s']  
        self.uid_t = config['src_tgt_pairs'][self.task]['uid_t']  
        self.iid_t = config['src_tgt_pairs'][self.task]['iid_t']  
        self.uid_all = config['src_tgt_pairs'][self.task]['uid']  #  181187
        self.iid_all = config['src_tgt_pairs'][self.task]['iid']  #  114495

        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src']  #  256
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt']  #  256
        self.batchsize_meta = config['src_tgt_pairs'][self.task]['batchsize_meta']  #  128
        self.batchsize_map = config['src_tgt_pairs'][self.task]['batchsize_map']  #  64
        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']  #  128
        self.batchsize_aug = self.batchsize_src  #  256

        self.input_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
            '/tgt_' + self.tgt + '_src_' + self.src  #  './data/ready/_8_2/tgt_CDs_and_Vinyl_src_Movies_and_TV'
        self.src_path = self.input_root + '/train_src.csv'
        self.tgt_path = self.input_root + '/train_tgt.csv'
        self.meta_path = self.input_root + '/train_meta.csv'
        self.test_path = self.input_root + '/test.csv'

        self.save_src_seq_path = self.input_root + '/user_interactions.pkl'

    
    def seq_extractor(self, x):  #  序列每个元素变成整型 并最后转换成array数组格式
            x = x.rstrip(']').lstrip('[').split(', ')  #  ['29752', '17954', '20722', '40344', '38292', '43851', '1332', '4671', '17272', '20894', '5320', '23034', '24257', '48242', ...]
            for i in range(len(x)):
                try:
                    x[i] = int(x[i])
                except:
                    x[i] = 0
            return np.array(x)

    def read_log_data_seqsrc(self, path, batchsize):
        '''源域每个用户的交互记录统计'''
        cols = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        data = pd.read_csv(path, header=None)
        data.columns = cols
        X = torch.tensor(data[x_col].values, dtype=torch.long)  #  torch.Size([1697533, 2])
        y = torch.tensor(data[y_col].values, dtype=torch.long)  #  torch.Size([1697533, 1])

        user_interactions = {}

        if not os.path.exists(self.save_src_seq_path):
            for index, row in data.iterrows():
                user_id = int(row['uid'])
                interaction = int(row['iid'])

                if user_id not in user_interactions:
                    user_interactions[user_id] = []

                # 限制每个用户的交互列表为最多20个交互
                if len(user_interactions[user_id]) < 20:
                    user_interactions[user_id].append(interaction)

            # 填充不足20个的交互
            for user_id, interactions in user_interactions.items():
                while len(interactions) < 20:
                    interactions.append(0)

            # 保存 user_interactions 字典到文件
            with open(self.save_src_seq_path, 'wb') as file:
                pickle.dump(user_interactions, file)

        else:
            with open(self.save_src_seq_path, 'rb') as file:
                user_interactions = pickle.load(file)
        
        if self.use_cuda:
                X = X.cuda()
                y = y.cuda()

        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, batchsize, shuffle=True)
        return data_iter, user_interactions

    
    def read_log_data_seqsrc_xy(self, path, batchsize):
        '''源域每个用户的交互记录统计'''
        cols = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        data = pd.read_csv(path, header=None)
        data.columns = cols
        X = torch.tensor(data[x_col].values, dtype=torch.long)  #  torch.Size([1697533, 2])
        y = torch.tensor(data[y_col].values, dtype=torch.long)  #  torch.Size([1697533, 1])

        user_interactions = {}

        if not os.path.exists(self.save_src_seq_path):
            for index, row in data.iterrows():
                user_id = int(row['uid'])
                interaction = int(row['iid'])

                if user_id not in user_interactions:
                    user_interactions[user_id] = []

                # 限制每个用户的交互列表为最多20个交互
                if len(user_interactions[user_id]) < 20:
                    user_interactions[user_id].append(interaction)

            # 填充不足20个的交互
            for user_id, interactions in user_interactions.items():
                while len(interactions) < 20:
                    interactions.append(0)

            # 保存 user_interactions 字典到文件
            with open(self.save_src_seq_path, 'wb') as file:
                pickle.dump(user_interactions, file)

        else:
            with open(self.save_src_seq_path, 'rb') as file:
                user_interactions = pickle.load(file)
        
        if self.use_cuda:
                X = X.cuda()
                y = y.cuda()

        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, batchsize, shuffle=True)
        return X, y, data_iter, user_interactions


    def read_log_data(self, path, batchsize, history=False):
        if not history:  #  没有历史记录
            cols = ['uid', 'iid', 'y']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)  #  torch.Size([1697533, 2])
            y = torch.tensor(data[y_col].values, dtype=torch.long)  #  torch.Size([1697533, 1])

        else:
            data = pd.read_csv(path, header=None)
            cols = ['uid', 'iid', 'y', 'pos_seq']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data.columns = cols
            pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=20, padding='post')
            pos_seq = torch.tensor(pos_seq, dtype=torch.long)  #  meta:torch.Size([374232, 20])    test:torch.Size([93866, 20])
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)  #  torch.Size([374232, 2])  torch.Size([93866, 2])
            X = torch.cat([id_fea, pos_seq], dim=1)  #  torch.Size([374232, 22])  torch.Size([93866, 22])
            y = torch.tensor(data[y_col].values, dtype=torch.long)  #  torch.Size([374232, 1])  torch.Size([93866, 1])
        

        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()

        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, batchsize, shuffle=True)
        return data_iter
    
    def countAverageScore(self):
        user_average_score = {}
        item_average_score = {}

        #  源域训练集
        cols = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        data = pd.read_csv(self.src_path, header=None)
        data.columns = cols
        X = torch.tensor(data[x_col].values, dtype=torch.long)  #  torch.Size([1697533, 2])
        y = torch.tensor(data[y_col].values, dtype=torch.long)  #  torch.Size([1697533, 1])

        for i in range(len(X)):
            user_id = X[i, 0].item()  
            rating = y[i].item()

            if user_id in user_average_score:
                user_average_score[user_id].append(rating)
            else:
                user_average_score[user_id] = [rating]
        

        user_avg_ratings = {}
        for user_id, ratings in user_average_score.items():
            avg_rating = 1.0 * sum(ratings) / len(ratings)
            user_avg_ratings[user_id] = avg_rating

        #  目标域训练集
        data = pd.read_csv(self.tgt_path, header=None)
        data.columns = cols
        X = torch.tensor(data[x_col].values, dtype=torch.long)  #  torch.Size([1697533, 2])
        y = torch.tensor(data[y_col].values, dtype=torch.long)  #  torch.Size([1697533, 1])

        for i in range(len(X)):
            item_id = X[i, 1].item()  # Assuming the item ID is in the second column (index 1)
            rating = y[i].item()

            if item_id in item_average_score:
                item_average_score[item_id].append(rating)
            else:
                item_average_score[item_id] = [rating]


        #  重叠用户 映射函数训练集
        cols = ['uid', 'iid', 'y', 'pos_seq']
        x_col = ['uid', 'iid']
        y_col = ['y']
        data = pd.read_csv(self.meta_path, header=None)
        data.columns = cols
        X = torch.tensor(data[x_col].values, dtype=torch.long)  #  torch.Size([1697533, 2])
        y = torch.tensor(data[y_col].values, dtype=torch.long)  #  torch.Size([1697533, 1])
           
        for i in range(len(X)):
            item_id = X[i, 1].item()  # Assuming the item ID is in the second column (index 1)
            rating = y[i].item()

            if item_id in item_average_score:
                item_average_score[item_id].append(rating)
            else:
                item_average_score[item_id] = [rating]


        item_avg_ratings = {}
        for item_id, ratings in item_average_score.items():
            avg_rating = 1.0 * sum(ratings) / len(ratings)
            item_avg_ratings[item_id] = avg_rating


        #  测试集
        cols = ['uid', 'iid', 'y', 'pos_seq']
        x_col = ['uid', 'iid']
        y_col = ['y']
        data = pd.read_csv(self.test_path, header=None)
        data.columns = cols
        X = torch.tensor(data[x_col].values, dtype=torch.long)  #  torch.Size([6971, 2])
        y = torch.tensor(data[y_col].values, dtype=torch.long)  #  torch.Size([6971, 1])

        return user_avg_ratings, item_avg_ratings, X, y     



    def read_map_data(self):
        cols = ['uid', 'iid', 'y', 'pos_seq']
        data = pd.read_csv(self.meta_path, header=None)  #  './data/ready/_8_2/xxx/train_meta.csv'
        data.columns = cols
        X = torch.tensor(data['uid'].unique(), dtype=torch.long)  #  meta训练集公共用户的数量 torch.Size([14425])
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)  #  tensor([    0,     1,     2,  ..., 14422, 14423, 14424])
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)  #  64 
        return data_iter

    def read_aug_data(self):
        cols_train = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train
        tgt = pd.read_csv(self.tgt_path, header=None)
        tgt.columns = cols_train

        X_src = torch.tensor(src[x_col].values, dtype=torch.long)  #  torch.Size([1697533, 2])
        y_src = torch.tensor(src[y_col].values, dtype=torch.long)  #  torch.Size([1697533, 1])
        X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)  #  torch.Size([1003726, 2])
        y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)  #  torch.Size([1003726, 1])
        X = torch.cat([X_src, X_tgt])  #  torch.Size([2701259, 2])
        y = torch.cat([y_src, y_tgt])  #  torch.Size([2701259, 1])
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)

        return data_iter

    
    def _lightgcn_adj_matrix(self, path):
        '''
        return: sparse adjacent matrix, refer lightgcn
        '''
        cols = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        data = pd.read_csv(path, header=None)
        data.columns = cols
        X = torch.tensor(data[x_col].values, dtype=torch.long)  #  torch.Size([1697533, 2])
        y = torch.tensor(data[y_col].values, dtype=torch.long)  #  torch.Size([1697533, 1])

        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        
        if path == self.src_path:
            self.training_user_s = X[:, 0].tolist()
            self.training_item_s = X[:, 1].tolist()
            self.num_user = self.uid_s
            self.num_node = self.uid_s + self.iid_s + 1

            user_np = np.array(self.training_user_s)
            item_np = np.array(self.training_item_s)
            ratings = np.ones_like(user_np, dtype=np.float32)

        elif path == self.tgt_path:
            self.training_user_t = X[:, 0].tolist()
            self.training_item_t = X[:, 1].tolist()
            self.num_user = self.uid_t
            self.num_node = self.uid_t + self.iid_t + 1
            
            user_np = np.array(self.training_user_t)
            item_np = np.array(self.training_item_t)
            ratings = np.ones_like(user_np, dtype=np.float32)
        

        #  item的序号要加上用户的数量
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_user)), shape=(self.num_node, self.num_node))
        adj_mat = tmp_adj + tmp_adj.T 
        
        #  pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # 创建稀疏矩阵的 coo_matrix 表示
        adj_matrix_coo = sp.coo_matrix(adj_matrix)

        # 转换为 PyTorch 的稀疏张量格式
        adj_matrix_sparse = torch.sparse.FloatTensor(
            torch.LongTensor([adj_matrix_coo.row, adj_matrix_coo.col]),
            torch.FloatTensor(adj_matrix_coo.data),
            torch.Size(adj_matrix_coo.shape)
        )
        if self.use_cuda:
            adj_matrix_sparse = adj_matrix_sparse.cuda()

        return adj_matrix_sparse