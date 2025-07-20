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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib

def randomforest_compute_cosine_similarity(cat_vector, domain_prototypes, device):
    # 将 cat_vector 转换为 PyTorch 张量，并移动到指定的设备
    cat_vector = cat_vector.clone().detach().unsqueeze(0).to(device) 
    # 将 domain_prototypes 转换为 PyTorch 张量，并移动到指定的设备
    domain_prototypes = torch.tensor(domain_prototypes, device=device)
    
    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(cat_vector.unsqueeze(1), domain_prototypes.unsqueeze(0), dim=2)  # (1, 15)
    
    return cosine_sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inputs: --model_path: path to model pth file;\
                                    --param_dict_pkl: the path to hyper-parameters;\
                                    --input: the path of input dataset(csv); \
                                    --output: output path of prediction result; \
                                    --has_label: whether the input dataset(csv) has labels;\
                                    --model_version:model and dict must be the same version' )

    parser.add_argument('--model_path', default = '/home/supermicro/code/AI4Sci/DLTKcat/data/model-xgl/proto_model/prot105_0.7724.pth')
    parser.add_argument('--prototypes_path', default = '/home/supermicro/code/AI4Sci/DLTKcat/data/model-xgl/proto_types/7724.pkl')
    parser.add_argument('--model_version', default = '/home/supermicro/code/AI4Sci/DLTKcat/data/pkl/prot_bert_dlt_mnk_sjh_mpek')
    parser.add_argument('--param_dict_pkl', default = '/home/supermicro/code/AI4Sci/DLTKcat/data/hyparams/param_2.pkl')
    parser.add_argument('--randomforest_model',default='/home/supermicro/code/AI4Sci/DLTKcat/data/model-xgl/randomforest/randomforest_model7724_randomstate42_R0.7853031728777188.pkl')
    parser.add_argument('--input', default = '/home/supermicro/code/AI4Sci/DLTKcat/cvFAP/RhKshA/RhKshA.csv')
    # parser.add_argument('--output', default = '/home/supermicro/code/AI4Sci/DLTKcat/cvFAP/RhKshA/RhKshA_pred.csv')
    parser.add_argument('--has_label', type=str, choices=['False','True'], default = 'True')
    # 预测数据csv生成的bert.pkl
    # bert_path = '/home/supermicro/code/AI4Sci/DLTKcat/bert/bert_dict_mpek.pkl'
    # bert_path = '/home/supermicro/code/AI4Sci/DLTKcat/bert/bert_dict_mnk2.pkl'

    args = parser.parse_args()

    #pred dict默认地址
    bert_path = '/home/supermicro/code/AI4Sci/DLTKcat/data/randomforest/bert/bert_dict.pkl'
    prot_path = '/home/supermicro/code/AI4Sci/DLTKcat/data/randomforest/prot/prot_dict.pkl'

    if os.path.isdir('../data/randomforest/bert'):
        os.system('rm -rf ../data/randomforest/bert')
    os.system('mkdir ../data/randomforest/bert')
    if os.path.isdir('../data/randomforest/prot'):
        os.system('rm -rf ../data/randomforest/prot')
    os.system('mkdir ../data/randomforest/prot')

    os.system('python /home/supermicro/code/AI4Sci/DLTKcat/bert_prot.py --data_path '+str(args.input)+' --output_path_bert '+bert_path+' --output_path_prot '+prot_path)
    
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
    if os.path.isdir('../data/randomforest/temp'):
        os.system('rm -rf ../data/randomforest/temp')
    os.system('mkdir ../data/randomforest/temp')

    os.system('python gen_features.py --data '+str(args.input)+' --output ../data/randomforest/temp/ --has_dict True \
                                                                              --has_label '+ str(args.has_label) +' --dict_path '+str(args.model_version)+'/dict' )
    data_input = load_data('../data/randomforest/temp/', has_label)
    
    predictions, labels = [], []
    batch_size = 16
    cat_vector_list = []
    randomforest_vector_list = []
    # failed_indices = []
    for i in range(math.ceil(len(data_input[0]) / batch_size)):

        batch_data = [ data_input[di][i * batch_size: (i + 1) * batch_size] for di in range(len(data_input))]
        if has_label:
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad,\
                                amino_mask, inv_Temp, Temp, label,smiles_names,seq_names = batch2tensor(batch_data, has_label, device)

        else:
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad,\
                                amino_mask, inv_Temp, Temp,smiles_names,seq_names = batch2tensor(batch_data, has_label, device)
        
        with torch.no_grad():
            pred , cat_vector = M( atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps, inv_Temp, Temp ,smiles_names,seq_names,epoch=0,label=None)
            for vector in cat_vector:
                cat_vector_list.append(vector)
        if has_label:
            labels += label.cpu().numpy().reshape(-1).tolist()

    labels = np.array(labels)

    domain_prototypes = torch.load( str( args.prototypes_path ) )
    
    for cat_vector in cat_vector_list:
        # 将 cat_vector 转换为 PyTorch 张量，并移动到指定的设备
        cat_vector_tensor = cat_vector.clone().detach().to(device)
        
        cosine_sim = randomforest_compute_cosine_similarity(cat_vector_tensor, domain_prototypes, device)
        
        cat_vector_numpy = cat_vector_tensor.cpu().numpy().reshape(1, -1)

        cat_vector_max = np.max(np.abs(cat_vector_numpy)) 
        cosine_sim_normalized = cosine_sim.cpu().numpy() * cat_vector_max / np.max(np.abs(cosine_sim.cpu().numpy()))
        
        randomforest_vector = np.concatenate((cosine_sim_normalized, cat_vector_numpy), axis=1)
  
        # randomforest_vector_list.append(randomforest_vector)
        randomforest_vector_list.append(cat_vector_numpy)

    model_filename = str(args.randomforest_model)
    rf_regressor = joblib.load(model_filename)
    print(f"Model loaded from {model_filename}")

    # 使用新的数据进行预测
    X = np.vstack(randomforest_vector_list)  
    y = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=26)

    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"R-squared: {r2}")

    model_filename = f"/home/supermicro/code/AI4Sci/DLTKcat/data/model_rsha/randomforest_model7724_randomstate42_R{r2}.pkl"
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)

    joblib.dump(rf_regressor, model_filename)

    # data = pd.read_csv( str(args.input) )
    # data['pred_log10kcat'] = y_pred
    # data['kcat'] = 10**data['pred_log10kcat']
    # data.to_csv(  str( args.output ) ,index=None )

    # print(f"Prediction task {str( args.output )} completed!")















