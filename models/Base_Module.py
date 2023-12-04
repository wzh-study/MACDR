import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_


def activation_layer(act_name):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
    
    Returns:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'tanh':
            act_layer = nn.Tanh()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_name.lower() == "softmax":
            act_layer = nn.Softmax(dim=1)
        elif act_name.lower() == 'leakyrelu':
            act_layer = nn.LeakyReLU()
        elif act_name.lower() == 'none':
            act_layer = None

    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError
    return act_layer

class MLP(nn.Module):
    """Multi Layer Perceptron Module, it is the most widely used module for 
    learning feature. Note we default add `BatchNorm1d` and `Activation` 
    `Dropout` for each `Linear` Module.

    Args:
        input dim (int): input size of the first Linear Layer.
        output_layer (bool): whether this MLP module is the output layer. If `True`, then append one Linear(*,1) module. 
        dims (list): output size of Linear Layer (default=[]).
        dropout (float): probability of an element to be zeroed (default = 0.5).
        activation (str): the activation function, support `[sigmoid, relu, prelu, dice, softmax]` (default='relu').

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)` or `(batch_size, dims[-1])`
    """

    def __init__(self, input_dim, output_layer=False, dims=None, dropout=0, activation="relu"):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(input_dim, i_dim))
            #layers.append(nn.BatchNorm1d(i_dim))
            if activation == 'softmax':
                layers.append(activation_layer(activation))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLPLayers(nn.Module):
    r""" MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(self, layers, dropout=0., activation='relu', bn=False, init_method=None):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)
    
    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == 'norm':
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)

class Pre_Train(torch.nn.Module):
    def __init__(self, data_config, uid_s, iid_s, uid_t, iid_t):
        super().__init__()
        self.latent_dim = data_config['latent_dim']  #  32
        
        self.src_model = MF(uid_s, iid_s, self.latent_dim)  
        self.tgt_model = MF(uid_t, iid_t, self.latent_dim)
    
    def forward(self, x, stage):
        if stage == 'train_src':  #  pre-train
                emb = self.src_model.forward(x)
                x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
                return x
            
        elif stage in ['train_tgt', 'test_tgt']:  #  only tgt
            emb = self.tgt_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)  #  torch.Size([256, 10]) * torch.Size([256, 10]) = torch.Size([256])
            return x

class MF(torch.nn.Module):
    def __init__(self, uid_all, iid_all, latent_dim):
        super().__init__()
        self.user_embeddings = torch.nn.Embedding(uid_all, latent_dim)  #  Embedding(181187, 10)
        self.item_embeddings = torch.nn.Embedding(iid_all + 1, latent_dim)  #  Embedding(114496, 10)

    def forward(self, x):  #  x:torch.Size([256, 2])
        uid_emb = self.user_embeddings(x[:, 0].unsqueeze(1))  #  torch.Size([256, 1, 10])
        iid_emb = self.item_embeddings(x[:, 1].unsqueeze(1))  #  torch.Size([256, 1, 10])
        emb = torch.cat([uid_emb, iid_emb], dim=1)  #  torch.Size([256, 2, 10])
        return emb
    



class LightGCNLayer(nn.Module):
    def __init__(self):
        super(LightGCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.sparse.mm(adj, embeds)

    
class LightGCN(nn.Module):
    def __init__(self, data_config, adj_matrix, mode='src'):
        super(LightGCN, self).__init__()

        if mode == 'src':
            self.num_user = data_config['src_tgt_pairs'][data_config['task']]['uid_s']
            self.num_item = data_config['src_tgt_pairs'][data_config['task']]['iid_s'] + 1
        else:
            self.num_user = data_config['src_tgt_pairs'][data_config['task']]['uid_t']
            self.num_item = data_config['src_tgt_pairs'][data_config['task']]['iid_t'] + 1

        self.latent_dim = data_config['latent_dim']

        self.gcn_layer = data_config['gcn_layer']
        self.gcnLayers = nn.Sequential(*[LightGCNLayer() for i in range(self.gcn_layer)])  #  2层
        
        #  init_Embedding
        user_emb_weight = self._weight_init(self.num_user, self.latent_dim, type='xa_uniform')
        item_emb_weight = self._weight_init(self.num_item, self.latent_dim, type='xa_uniform')
        
        self.user_embeddings = nn.Embedding(self.num_user, self.latent_dim, _weight=user_emb_weight)
        self.item_embeddings = nn.Embedding(self.num_item, self.latent_dim, _weight=item_emb_weight)

        #  adj_matrix
        self.adj_matrix = adj_matrix

   
    #  初始化 embedding矩阵参数 即用户和物品的embedding矩阵
    def _weight_init(self, rows, cols, type='xa_norm'):
        '''
        nn.init.xavier_normal_ : 使用均值为0,标准差为$\sqrt{\frac{2}{n_{in}+n_{out}}}$的正态分布来初始化权重，其中$n_{in}$和$n_{out}$分别是权重张量的输入和输出通道数。这个方法适用于激活函数为tanh的情况。

        nn.init.normal_: 使用均值为0,标准差为std的正态分布来初始化权重。其中,std可以通过指定std参数来控制。

        nn.init.xavier_uniform_: 使用均匀分布$U[-a, a]$来初始化权重,其中$a=\sqrt{\frac{6}{n_{in}+n_{out}}}$。这个方法适用于激活函数为ReLU的情况。

        需要注意的是,这些初始化方法一般只用于初始化权重,对于偏置一般可以直接初始化为0
        '''
        if type == 'norm':
            free_emb = nn.init.normal_(torch.empty(rows, cols), std=0.01)
        elif type == 'xa_norm':
            free_emb = nn.init.xavier_normal_(torch.empty(rows, cols))
        elif type == 'xa_uniform':
            free_emb = nn.init.xavier_uniform_(torch.empty(rows, cols))
        return free_emb
    
    def getEgoEmbeds(self):
        return torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)

    def forward(self):
        ego_emb = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_emb = [ego_emb]

        #  Graph convolution
        for _, gcn in enumerate(self.gcnLayers):
            tmp_emb = gcn(self.adj_matrix, all_emb[-1])
            all_emb.append(tmp_emb)

        #  Fusion
        all_emb = torch.stack(all_emb, dim=1)
        mean_emb = torch.mean(all_emb, dim=1)

        #  Split_emb 
        user_emb, item_emb = torch.split(mean_emb, [self.num_user, self.num_item])
        return user_emb, item_emb
    
    
