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
from train_functions import batch2tensor, load_data, scores
import os
import warnings
from DLTKcat import DLTKcat

def get_domain(label):
    return int(np.floor(label)) 

def process_data(features, labels):
    df = pd.DataFrame({
        'smoothed_cat_vector': features,
        'label': labels
    })
    
    df['domain'] = df['label'].apply(get_domain)

    domain_means = df.groupby('domain')['smoothed_cat_vector'].apply(lambda x: np.mean(np.vstack(x), axis=0))
    
    domain_label_means = df.groupby('domain')['label'].mean()
    
    domain_summary = pd.DataFrame({
        'feature_vector_mean': domain_means.values,
        'label_mean': domain_label_means.values
    })

    domain_summary['feature_vector_mean'] = domain_summary['feature_vector_mean'].apply(lambda x: np.array2string(x, separator=',')[1:-1])
    if os.path.isdir('../proto/prototypes'):
        os.system('rm -rf ../proto/prototypes')
    os.system('mkdir ../proto/prototypes')
   
    domain_summary.to_pickle(str(args.output))
    print("Domain summary has been saved to domain_summary.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inputs: --model_path: path to model pth file;\
                                    --param_dict_pkl: the path to hyper-parameters;\
                                    --input: the path of input dataset(csv); \
                                    --output: output path of prediction result; \
                                    --has_label: whether the input dataset(csv) has labels;\
                                    --model_version:model and dict must be the same version' )

    parser.add_argument('--model_path', required = True)
    parser.add_argument('--model_version', required = True)
    parser.add_argument('--param_dict_pkl', default = '/home/supermicro/code/AI4Sci/DLTKcat/data/hyparams/param_2.pkl')
    parser.add_argument('--input', required = True)
    parser.add_argument('--output', required = True)
    parser.add_argument('--has_label', type=str, choices=['False','True'], default = 'True')
    # 预测数据csv生成的bert.pkl
    # bert_path = '/home/supermicro/code/AI4Sci/DLTKcat/bert/bert_dict_mpek.pkl'
    # bert_path = '/home/supermicro/code/AI4Sci/DLTKcat/bert/bert_dict_mnk2.pkl'

    args = parser.parse_args()

    # if os.path.isdir('../data/pred/bert'):
    #     os.system('rm -rf ../data/pred/bert')
    # os.system('mkdir ../data/pred/bert')
    
    # if os.path.isdir('../data/pred/prot'):
    #     os.system('rm -rf ../data/pred/prot')
    # os.system('mkdir ../data/pred/prot')

    # os.system('python /home/supermicro/code/AI4Sci/DLTKcat/bert_prot.py --data_path '+str(args.input))
    
    bert_path = '/home/supermicro/code/AI4Sci/DLTKcat/data/pred/bert/bert_dict.pkl'
    prot_path = '/home/supermicro/code/AI4Sci/DLTKcat/data/pred/prot/prot_dict.pkl'

    if str(args.has_label) == 'False':
        has_label = False
    else:
        has_label = True

    param_dict = load_pickle( str( args.param_dict_pkl ) )
    atom_dict = load_pickle( str( args.model_version )+ '/dict/fingerprint_dict.pkl' )
    word_dict = load_pickle( str( args.model_version )+ '/dict/word_dict.pkl' )
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU!')
    else:
        device = torch.device('cpu')
        print('CPU!')

    print('Task '+ str(args.input)+' started!')    

    comp_dim, prot_dim, gat_dim, num_head, dropout, alpha, window, layer_cnn, latent_dim, layer_out = \
                      param_dict['comp_dim'], param_dict['prot_dim'],param_dict['gat_dim'],param_dict['num_head'],\
                      param_dict['dropout'], param_dict['alpha'], param_dict['window'], param_dict['layer_cnn'], \
                      param_dict['latent_dim'], param_dict['layer_out']

    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
    # Load model
    M = DLTKcat( len(atom_dict), len(word_dict), comp_dim, prot_dim, gat_dim, num_head, \
                                        dropout, alpha, window, layer_cnn, latent_dim, layer_out,bert_path,prot_path)
    M.to(device);
    M.load_state_dict(torch.load( str( args.model_path ), map_location=device  ))
    # Prep input
    # if os.path.isdir('../data/pred/temp'):
    #     os.system('rm -rf ../data/pred/temp')

    # os.system('mkdir ../data/pred/temp')
    # os.system('python gen_features.py --data '+str(args.input)+' --output ../data/pred/temp/ --has_dict True \
    #                                                                           --has_label '+ str(args.has_label) +' --dict_path '+str(args.model_version)+'/dict' )
    data_input = load_data('/home/supermicro/code/AI4Sci/DLTKcat/proto/temp', has_label)
    
    labels,features = [], []
    batch_size = 16
    # failed_indices = []
    for i in range(math.ceil(len(data_input[0]) / batch_size)):
        # print('i:',i)
        batch_data = [ data_input[di][i * batch_size: (i + 1) * batch_size] for di in range(len(data_input))]
        
        if has_label:
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad,\
                                amino_mask, inv_Temp, Temp, label,smiles_names,seq_names = batch2tensor(batch_data, has_label, device)
        else:
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad,\
                                amino_mask, inv_Temp, Temp,smiles_names,seq_names = batch2tensor(batch_data, has_label, device)
        
        with torch.no_grad():
            pred,smoothed_cat_vector = M( atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps, inv_Temp, Temp ,smiles_names,seq_names,epoch=0,label=None)

        smoothed_cat_vector = smoothed_cat_vector.cpu().detach().numpy()  # shape: (batch_size, 122)
        label = label.cpu().numpy().reshape(-1)  

        for j in range(smoothed_cat_vector.shape[0]):  
            smoothed_cat_vector_flattened = smoothed_cat_vector[j, :]  
            features.append(smoothed_cat_vector_flattened)  
            labels.append(label[j])  

    #Delete intermediate files
    os.system('rm -rf ../data/pred/temp')

    print('Feature generation completed for', str(args.input))
    process_data(features, labels)

    print('Task '+ str(args.input)+' completed!')















