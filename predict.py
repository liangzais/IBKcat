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
import shap
import matplotlib.pyplot as plt

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
    parser.add_argument('--bert', required = True)
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
    
    # #pred dict默认地址
    # bert_path = '/home/supermicro/code/AI4Sci/DLTKcat/data/pred/bert/bert_dict.pkl'
    # prot_path = '/home/supermicro/code/AI4Sci/DLTKcat/data/pred/prot/prot_dict.pkl'
    bert_path = str( args.bert )
    prot_path = bert_path

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
    if os.path.isdir('../data/pred/temp'):
        os.system('rm -rf ../data/pred/temp')

# python gen_features.py --data /home/supermicro/code/AI4Sci/DLTKcat/data/pkl/bert_dlt_mnk_sjh_mpek/all_data.csv --output /home/supermicro/code/AI4Sci/DLTKcat/data/pkl/bert_dlt_mnk_sjh_mpek/all --has_dict False --dict_path /home/supermicro/code/AI4Sci/DLTKcat/data/pkl/bert_dlt_mnk_sjh_mpek/dict 
    os.system('mkdir ../data/pred/temp')
    os.system('python gen_features.py --data '+str(args.input)+' --output ../data/pred/temp/ --has_dict True \
                                                                              --has_label '+ str(args.has_label) +' --dict_path '+str(args.model_version)+'/dict' )
    data_input = load_data('../data/pred/temp/', has_label)
    
    predictions, labels = [], []
    cat_vec_list = []
    batch_size = 16
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
            pred,cat_vec = M( atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps, inv_Temp, Temp ,smiles_names,seq_names,epoch=0,label=None)
            cat_vec_list.append(cat_vec.cpu().detach().numpy())
        predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
        if has_label:
            labels += label.cpu().numpy().reshape(-1).tolist()

#     feature_vectors = np.concatenate(cat_vec_list, axis=0)

#     feature_vectors_tensor = torch.tensor(feature_vectors, dtype=torch.float32).to('cuda')
#     def forward_shap(cat_vector):
#         M.eval()
#         with torch.no_grad():
#             # 将 numpy.ndarray 转换为 torch.Tensor
#             cat_vector_tensor = torch.tensor(cat_vector, dtype=torch.float32).to(device)
#             return M.output(cat_vector_tensor).cpu().numpy()

#     # 选择背景数据（例如前10个样本）
#     background_data = feature_vectors[:1000]

#     # 初始化 Explainer
#     explainer = shap.Explainer(forward_shap, background_data)

#     # 计算 SHAP 值
#     shap_values = explainer(feature_vectors)

#     shap_values_array = shap_values.values  # SHAP 值数组
#     feature_vectors_array = feature_vectors  # 特征矩阵

# # 将 SHAP 值和特征矩阵保存为一个字典
#     shap_data = {
#         "shap_values": shap_values_array,
#         "feature_vectors": feature_vectors_array
#     }

# # 保存为 .pkl 文件
#     output_shap_path = "/home/supermicro/code/AI4Sci/DLTKcat/plot/shap_data.pkl"
#     with open(output_shap_path, "wb") as f:
#         pickle.dump(shap_data, f)

#     print(f"SHAP data saved to {output_shap_path}")

    predictions = np.array(predictions)

    labels = np.array(labels)
    if has_label:
        rmse, r2 = scores(labels, predictions)
        print('Accuracy: RMSE='+str(rmse)+', R2='+str(r2) )
    else:
        print('No labels provided.')

    #Save prediction results
    predictions = predictions.reshape(-1) 
    data = pd.read_csv( str(args.input) )
    data['pred_log10kcat'] = predictions
    data['kcat'] = 10**data['pred_log10kcat']
    data.to_csv(  str( args.output ) ,index=None )
    #Delete intermediate files
    os.system('rm -rf ../data/pred/temp')
    print('Task '+ str(args.input)+' completed!')















