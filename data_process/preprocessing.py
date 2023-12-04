import pandas as pd
import gzip
import json
import tqdm
import random
import os

class DataPreprocessingMid():
    def __init__(self,
                 root,
                 dealing):
        self.root = root
        self.dealing = dealing

    def main(self):
        print('Parsing ' + self.dealing + ' Mid...')
        re = []
        with gzip.open(self.root + 'raw/reviews_' + self.dealing + '_5.json.gz', 'rb') as f:
            for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
                line = json.loads(line)  #  {'reviewerID': 'A10000012B7CGYKOMPQ4L', 'asin': '000100039X', 'reviewerName': 'Adam', 'helpful': [0, 0], 'reviewText': 'Spiritually and ment...eally are!', 'overall': 5.0, 'summary': 'Wonderful!', 'unixReviewTime': 1355616000, 'reviewTime': '12 16, 2012'}
                re.append([line['reviewerID'], line['asin'], line['overall']])  #  ['A10000012B7CGYKOMPQ4L', '000100039X', 5.0]
        re = pd.DataFrame(re, columns=['uid', 'iid', 'y'])
        print(self.dealing + ' Mid Done.')
        re.to_csv(self.root + 'mid/' + self.dealing + '.csv', index=0)
        return re

class DataPreprocessingReady():
    def __init__(self,
                 root,
                 src_tgt_pairs,
                 task,
                 ratio):
        self.root = root  #  './data/'
        self.src = src_tgt_pairs[task]['src']  #  'Sports_and_Outdoors'
        self.tgt = src_tgt_pairs[task]['tgt']  #  'Clothing_Shoes_and_Jewelry'
        self.ratio = ratio  #  [0.8, 0.2]

    def read_mid(self, field):
        path = self.root + 'mid/' + field + '.csv'
        re = pd.read_csv(path)
        return re

    def mapper(self, src, tgt):
        print('Source inters: {}, uid: {}, iid: {}, Density: {}.'.format(len(src), len(set(src.uid)), len(set(src.iid)), len(src) / (len(set(src.uid)) * len(set(src.iid)) )))  #  Source inters: 296337, uid: 35598, iid: 18357, Density: 0.00045348045456958995.
        print('Target inters: {}, uid: {}, iid: {}, Density: {}.'.format(len(tgt), len(set(tgt.uid)), len(set(tgt.iid)), len(tgt) / (len(set(tgt.uid)) * len(set(tgt.iid)) )))  #  Target inters: 278677, uid: 39387, iid: 23033, Density: 0.0003071833809100676.
        co_uid = set(src.uid) & set(tgt.uid)  #  域重叠用户  {'A38TVIYFEGSBWW', 'A3T5FHZQQBBLM2', 'A1Y0VQV0SRODGP', 'A3OJSXR0NPFDNJ', 'A15LDYHVYWRN9Z', 'A3PU4D7YKP5KWG', 'A3775OP5VTX5ON', 'A37FCNU3EY6LZM', 'A2WGD7ZK0MBS7S', 'AD64M1IOZBT0N', 'A12RNPBA6RHPIG', 'AQ2FC1DLKVD8H', 'A372A2682IXU12', 'A3SQV1Q3LQH59Z', ...}
        all_uid = set(src.uid) | set(tgt.uid)  #  两个域的总用户
        print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))  #  All uid: 71077, Co uid: 3908.
        
        #  创建重叠用户的映射
        uid_dict_co = dict(zip(co_uid, range(len(co_uid))))  #  len(): 3908
        share_user_num = len(co_uid)  #  len(): 3908

        #  创建物品的映射  从1开始  padding为0
        iid_dict_src = dict(zip(set(src.iid), range(1, len(set(src.iid)) + 1)))
        iid_dict_tgt = dict(zip(set(tgt.iid), range(1, len(set(tgt.iid)) + 1)))

        # 将未重叠用户映射到重叠用户后的ID范围
        src_non_co_uid = set(src.uid) - co_uid
        tgt_non_co_uid = set(tgt.uid) - co_uid
        
        src_non_co_uid_dict = dict(zip(src_non_co_uid, range(share_user_num, share_user_num + len(src_non_co_uid))))
        tgt_non_co_uid_dict = dict(zip(tgt_non_co_uid, range(share_user_num, share_user_num + len(tgt_non_co_uid))))

        # 合并映射字典
        src_uid_mapping = {**uid_dict_co, **src_non_co_uid_dict}
        tgt_uid_mapping = {**uid_dict_co, **tgt_non_co_uid_dict}

        # 更新源域和目标域的用户和物品映射
        src.uid = src.uid.map(src_uid_mapping)  #  min=0  max(src.uid)=35597
        tgt.uid = tgt.uid.map(tgt_uid_mapping)  #  min=0  max=39386
        src.iid = src.iid.map(iid_dict_src)  #  18356
        tgt.iid = tgt.iid.map(iid_dict_tgt)  #  23032

        return src, tgt, share_user_num  #  重叠用户再前

    def get_history(self, data, uid_set):
        pos_seq_dict = {}
        for uid in tqdm.tqdm(uid_set):  #  对每个公共用户
            pos = data[(data.uid == uid) & (data.y > 3)].iid.values.tolist()  #  分数>3的用户交互过的物品为正样本
            pos_seq_dict[uid] = pos  #  源域中每个用户交互过的一些正样本对
        return pos_seq_dict  

    def split(self, src, tgt, share_user_num):
        print('All iid: {}.'.format(len(set(src.iid)) + len(set(tgt.iid))) )  #  All iid: 41390.
        src_users = set(src.uid.unique())  #  35598
        tgt_users = set(tgt.uid.unique())  #  39387
        co_users = set(range(share_user_num))  #  3908  {0, 1, 2, ...}
        test_users = set(random.sample(co_users, round(self.ratio[1] * len(co_users))))  #  选择公共用户中的20%作为test  782
        train_src = src  #  (296337, 3)
        train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]  #  (271675, 3)  除去uid为test中公共用户的 其他目标域用户为目标域训练集
        test = tgt[tgt['uid'].isin(test_users)]  #  目标域测试集里被选中的公共用户交互  (7402, 3)
        pos_seq_dict = self.get_history(src, co_users)  #  每个公共用户的正样本交互序列 3908
        train_meta = tgt[tgt['uid'].isin(co_users - test_users)]  #  (27713, 3)  目标域中其余非测试集公共用户(80%)当作meta train
        train_meta['pos_seq'] = train_meta['uid'].map(pos_seq_dict)  #  加一列  (27713,)  每个用户交互的正样本数量不一致
        test['pos_seq'] = test['uid'].map(pos_seq_dict)  #  (7002, 4)
        return train_src, train_tgt, train_meta, test

    def save(self, train_src, train_tgt, train_meta, test):
        output_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
                      '/tgt_' + self.tgt + '_src_' + self.src
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        print(output_root)
        train_src.to_csv(output_root + '/train_src.csv', sep=',', header=None, index=False)
        train_tgt.to_csv(output_root + '/train_tgt.csv', sep=',', header=None, index=False)
        train_meta.to_csv(output_root +  '/train_meta.csv', sep=',', header=None, index=False)
        test.to_csv(output_root + '/test.csv', sep=',', header=None, index=False)

    def main(self):
        src = self.read_mid(self.src)  #  (296337, 3)   读取mid文件   uid  iid  y
        tgt = self.read_mid(self.tgt)  #  (278677, 3)
        src, tgt, share_user_num = self.mapper(src, tgt)  #  统计重叠用户和总用户数 将物品id 进行映射成数字id
        train_src, train_tgt, train_meta, test = self.split(src, tgt, share_user_num)  #  划分训练用的几个数据集  源域训练集：用全部源域数据  目标域训练集：用除去20%的公共用户的其余目标域用户测试集  meta训练集：重叠用户中不在测试集中的20%重叠用户对应的目标域交互、评分、源域交互序列  test测试集:20%公共用户的目标域交互、评分、对应用户的源域交互序列  
        self.save(train_src, train_tgt, train_meta, test)  #  保存文件

