import os
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

from utils.set import *

from run_main.run_EMCDR import Run as run_EMCDR
from run_main.run_PTUPCDR import Run as run_PTUPCDR
from run_main.run_MACDR import Run as MACDR


def prepare(config_path):
    parser = argparse.ArgumentParser()
    #  PreProcess Data
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=0)
    parser.add_argument('--task', default='1')
    parser.add_argument('--runid', default='0')  
    #  Pre-train Model and Cross Model
    parser.add_argument('--base_model', default='MF')  
    parser.add_argument('--cross_model', default='EMCDR')  
    #  Generator Parameter
    parser.add_argument('--num_fields', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--meta_dim', type=int, default=50)
    parser.add_argument('--ratio', default=[0.2, 0.8])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--pre_train_epoch', type=int, default=30)  
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01) 
    parser.add_argument('--lr_meta', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0)

    #  IntentCDR 
    parser.add_argument('--margin', type=int, default=0.2)  
    parser.add_argument('--source_domain_reg', type=float, default=0.01)  
    parser.add_argument('--lamda', type=float, default=200)  
    parser.add_argument('--neg_sample', type=int, default=10)  
    parser.add_argument('--dropout', type=float, default=0.8) 

    parser.add_argument('--num_cluster', type=int, default=20)  
    parser.add_argument('--closest_center', type=int, default=1)   
    parser.add_argument('--gcn_layer', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--n_task', type=int, default=2)
    parser.add_argument('--n_expert', type=int, default=2) 
    parser.add_argument('--n_mlplayer', type=int, default=4)   

    args = parser.parse_args()
    
    seed_everything(args.seed)

    with open(config_path, 'r') as f:
        config = json.load(f)
        config['runid'] = args.runid
        config['base_model'] = args.base_model  #  'MF'
        config['task'] = args.task  #  '1'  
        config['num_fields'] = args.num_fields
        config['latent_dim'] = args.latent_dim
        config['meta_dim'] = args.meta_dim
        config['ratio'] = args.ratio  #  [0.8, 0.2]
        config['pre_train_epoch'] = args.pre_train_epoch    #  Task1:100  Task2:30  Task3:30
        config['epoch'] = args.epoch  #  100
        config['lr'] = args.lr  #  0.01
        config['lr_meta'] = args.lr_meta
        config['weight_decay'] = args.weight_decay
        config['num_cluster'] = args.num_cluster
        config['closest_center'] = args.closest_center
        config['gcn_layer'] = args.gcn_layer
        config['temperature'] = args.temperature 
        config['source_domain_reg'] = args.source_domain_reg

        config['dropout'] = args.dropout
        config['margin'] = args.margin
        config['lamda'] = args.lamda
        config['l2_reg'] = args.l2_reg
        config['neg_sample'] = args.neg_sample
        config['n_task'] = args.n_task
        config['n_expert'] = args.n_expert  
        config['n_mlplayer'] = args.n_mlplayer

    return args, config


if __name__ == '__main__':
    config_path = './config.json'
    args, config = prepare(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  #  '0'

    print('task:{}; model:{}; ratio:{}; epoch:{}; lr:{}; gpu:{}; seed:{};'.
          format(args.task, args.base_model, args.ratio, args.epoch, args.lr, args.gpu, args.seed))  #  task:1; model:MF; ratio:[0.8, 0.2]; epoch:10; lr:0.01; gpu:0; seed:42;

    if not args.process_data_mid and not args.process_data_ready:
        if args.cross_model == "EMCDR":
            run_EMCDR(config).main()
        elif args.cross_model == "PTUPCDR":
            run_PTUPCDR(config).main()
        elif args.cross_model == "MACDR":
            run_MACDR(config).main()
        