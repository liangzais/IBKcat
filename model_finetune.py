import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
from scipy import stats
import pickle
import argparse
import math
from math import sqrt
import numpy as np
import pandas as pd
from feature_functions import load_pickle
import torch.nn.functional as F
from torch.autograd import Variable
from train_functions import *
from train_functions import batch2tensor, load_data, scores
import os
import warnings
from DLTKcat import DLTKcat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inputs: --model_path: path to model pth file;\
                                    --param_dict_pkl: the path to hyper-parameters;\
                                    --input: the path of input dataset(csv); \
                                    --model_version:model and dict must be the same version' )

    parser.add_argument('--model_path', default = '/home/supermicro/code/AI4Sci/DLTKcat/data/model-xgl/proto_model/prot105_0.7724.pth')
    parser.add_argument('--model_version',  default = '/home/supermicro/code/AI4Sci/DLTKcat/data/pkl/prot_bert_dlt_mnk_sjh_mpek')
    parser.add_argument('--param_dict_pkl', default = '/home/supermicro/code/AI4Sci/DLTKcat/data/hyparams/param_2.pkl')
    parser.add_argument('--ft_version', default='/home/supermicro/code/AI4Sci/DLTKcat/data/pkl/prototypes')
    # 初始学习率0.001
    parser.add_argument('--lr', default = 0.0001, type=float )
    parser.add_argument('--batch', default = 32 , type=int )
    parser.add_argument('--lr_decay', default = 0.5, type=float )
    parser.add_argument('--decay_interval', default = 10, type=int )
    parser.add_argument('--num_epoch', default = 40, type=int )
    parser.add_argument('--prototypes_path', default='/home/supermicro/code/AI4Sci/DLTKcat/data/model-xgl/proto_types/7724.pkl', type=str)

    args = parser.parse_args()

    lr, batch_size, lr_decay, decay_interval, param_dict_pkl=float(args.lr), int(args.batch), float(args.lr_decay), int(args.decay_interval) , str( args.param_dict_pkl )
    num_epochs = int( args.num_epoch )
    all_data = str(args.ft_version) + '/all_data.csv'
    train_path = str(args.ft_version) + '/train'
    test_path = str(args.ft_version) + '/test'
    bert_path = str(args.ft_version) + '/bert_all_data.pkl'
    prot_path = str(args.ft_version) + '/prot_all_data.pkl'

    # os.system('python /home/supermicro/code/AI4Sci/DLTKcat/bert_prot.py --data_path '+all_data+' --output_path_bert '+bert_path+' --output_path_prot '+ prot_path)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU!')
    else:
        device = torch.device('cpu')
        print('CPU!')

    param_dict = load_pickle( str( args.param_dict_pkl ) )
    atom_dict = load_pickle( str( args.model_version )+ '/dict/fingerprint_dict.pkl' )
    word_dict = load_pickle( str( args.model_version )+ '/dict/word_dict.pkl' )
    
    datapack = load_data(train_path, True )
    test_data = load_data(test_path, True )
    train_data, dev_data = split_data( datapack, 0.1 )
    domain_prototypes = torch.load( str( args.prototypes_path ) )

    comp_dim, prot_dim, gat_dim, num_head, dropout, alpha, window, layer_cnn, latent_dim, layer_out = \
                      param_dict['comp_dim'], param_dict['prot_dim'],param_dict['gat_dim'],param_dict['num_head'],\
                      param_dict['dropout'], param_dict['alpha'], param_dict['window'], param_dict['layer_cnn'], \
                      param_dict['latent_dim'], param_dict['layer_out']

    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
    # Load model
    M = DLTKcat( len(atom_dict), len(word_dict), comp_dim, prot_dim, gat_dim, num_head, \
                                        dropout, alpha, window, layer_cnn, latent_dim, layer_out,bert_path,prot_path)
    M.to(device)

    M.load_state_dict(torch.load( str( args.model_path ), map_location=device  ))

    rmse_train_scores, r2_train_scores, rmse_test_scores, r2_test_scores, rmse_dev_scores, r2_dev_scores = \
        finetune_eval(M, train_data, test_data, dev_data, device, lr, batch_size, lr_decay, decay_interval, num_epochs, domain_prototypes)
        
    epoch_inds = list(np.arange(1, num_epochs + 1))
    result = pd.DataFrame(zip(epoch_inds, rmse_train_scores, r2_train_scores, rmse_test_scores, r2_test_scores, \
                               rmse_dev_scores, r2_dev_scores), \
                         columns=['epoch', 'RMSE_train', 'R2_train', 'RMSE_test', 'R2_test', 'RMSE_dev', 'R2_dev'])
    
    output_path = os.path.join('../data/performances/', os.path.basename(param_dict_pkl).split('.')[0] + \
                               '_lr=' + str(lr) + '_batch=' + str(batch_size) + \
                               '_lr_decay=' + str(lr_decay) + '_decay_interval=' + str(decay_interval) + \
                               '_shuffle_T=False' + '.csv')
    result.to_csv(output_path, index=None)
    print('Done for ' + param_dict_pkl + '.')







