import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from FDS.fds import FDS

config = dict(feature_dim=122, start_update=0, start_smooth=1, kernel='gaussian', ks=5, sigma=2)
def batch_pad_bert(arr):
    '''
    Pad feature vectors all into the same length.
    '''
    N = max([a.shape[0] for a in arr])
    if arr[0].ndim == 1:
        new_arr = np.zeros((len(arr), N, arr[0].shape[1]))
        new_arr_mask = np.zeros((len(arr), N))
        for i, a in enumerate(arr):
            n = a.shape[0]
            new_arr[i, :n, :] = a
            new_arr_mask[i, :n] = 1
        return new_arr

    elif arr[0].ndim == 2:
        max_dim = max([a.shape[1] for a in arr])
        new_arr = np.zeros((len(arr), N, max_dim))
        new_arr_mask = np.zeros((len(arr), N, max_dim))
        for i, a in enumerate(arr):
            n, dim = a.shape
            new_arr[i, :n, :dim] = a
            new_arr_mask[i, :n, :dim] = 1
        return new_arr

def batch_pad_seq(arr):
    '''
    Pad feature vectors all into the same length using PyTorch tensors.
    '''
    N = max(a.shape[0] for a in arr)  # 假设arr是张量列表
    M = max([a.shape[1] for a in arr])
    new_arr = torch.zeros((len(arr), N, M), device=arr[0].device)  # 使用相同的设备
    new_arr_mask = torch.zeros((len(arr), N, M), device=arr[0].device)
    for i, a in enumerate(arr):
        n = a.shape[0]
        new_arr[i, :n, :] = a
        new_arr_mask[i, :n, :] = 1  
    return new_arr, new_arr_mask


class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(3), self.alpha)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)

        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        b = Wh.size()[0]
        N = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat_interleave(N, dim=0).view(b, N*N, self.out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)

        return all_combinations_matrix.view(b, N, N, 2 * self.out_features)


class DLTKcat(nn.Module):
    def __init__(self, n_atom, n_amino, comp_dim, prot_dim, gat_dim, num_head, dropout, alpha, window, layer_cnn, latent_dim, layer_out,bert_path,prot_path ):
        super(DLTKcat, self).__init__()
        '''
        n_atom here stands for number of atom_features
        '''
        # {'comp_dim': 80, 'prot_dim': 80, 'gat_dim': 50, 'num_head': 3, 'dropout': 0.1, 'alpha': 0.1, 'window': 5, 'layer_cnn': 4, 'latent_dim': 40, 'layer_out': 4}
        # here we need to ues llm to embedd the smiles
        # n_atom+1 = fingerprints length + 1 
        self.embedding_layer_atom = nn.Embedding(n_atom+1, comp_dim)
        # bert_path = '/home/supermicro/code/AI4Sci/DLTKcat/bert_dict.pkl'

        with open(bert_path, 'rb') as file:
            bert_dict = pickle.load(file)
        self.saved_embedding_dict = bert_dict

        # prot_path = '/home/supermicro/code/AI4Sci/DLTKcat/prot_dict.pkl'
        with open(prot_path, 'rb') as file:
            prot_dict = pickle.load(file)
        self.saved_embedding_prot = prot_dict

        self.prot_com_dim = nn.Linear(1024,40)

        self.bert_com_dim = nn.Linear(300,comp_dim)
        self.embedding_layer_amino = nn.Embedding(n_amino+1, prot_dim)

        self.Fds = FDS(**config)

        self.n_atom = n_atom
        self.comp_dim = comp_dim

        self.dropout = dropout
        self.alpha = alpha
        self.layer_cnn = layer_cnn
        self.latent_dim = latent_dim
        self.layer_out = layer_out

        self.gat_layers = [GATLayer(comp_dim, gat_dim, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(num_head)]
        for i, layer in enumerate(self.gat_layers):
            self.add_module('gat_layer_{}'.format(i), layer)
        self.gat_out = GATLayer(gat_dim * num_head, comp_dim, dropout=dropout, alpha=alpha, concat=False)
        self.W_comp = nn.Linear(comp_dim, latent_dim)

        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*window+1,
                                                    stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_prot = nn.Linear(prot_dim, latent_dim)

        self.fp0 = nn.Parameter(torch.empty(size=(1024, latent_dim)))
        nn.init.xavier_uniform_(self.fp0, gain=1.414)
        self.fp1 = nn.Parameter(torch.empty(size=(latent_dim, latent_dim)))
        nn.init.xavier_uniform_(self.fp1, gain=1.414)

        self.bidat_num = 4

        self.U = nn.ParameterList([nn.Parameter(torch.empty(size=(latent_dim, latent_dim))) for _ in range(self.bidat_num)])
        for i in range(self.bidat_num):
            nn.init.xavier_uniform_(self.U[i], gain=1.414)

        self.transform_c2p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.transform_p2c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])

        self.bihidden_c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.bihidden_p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.biatt_c = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])
        self.biatt_p = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])

        self.comb_c = nn.Linear(latent_dim * self.bidat_num, latent_dim)
        self.comb_p = nn.Linear(latent_dim * self.bidat_num, latent_dim)
   
        self.W_out = nn.ModuleList([nn.Linear(latent_dim * 3 + 2, latent_dim * 3 + 2)
                                    for _ in range(self.layer_out)])
    
        self.output = nn.Linear(latent_dim * 3 + 2, 1)
        
    def comp_gat(self, atoms_vector, adj):
        # print(atoms)
        # atoms_vector.shape  4 n 70
        # atoms_vector = self.embedding_layer_atom(atoms)
        # print('shape',atoms_vector.shape)
        atoms_multi_head = torch.cat([gat(atoms_vector, adj) for gat in self.gat_layers], dim=2)
        atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj))
        atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha)
        return atoms_vector
    # def comp_gat1(self, atoms, adj):
    #     atoms_vector = self.embedding_layer_atom(atoms)
    #     atoms_multi_head = torch.cat([gat(atoms_vector, adj) for gat in self.gat_layers], dim=2)
    #     atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj))
    #     atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha)
    #     return atoms_vector

    def prot_cnn(self, amino ):
        amino_vector = self.embedding_layer_amino(amino)
        amino_vector = torch.unsqueeze(amino_vector, 1)
        # print('amino_vector1:',amino_vector.shape)
        for i in range(self.layer_cnn):
            amino_vector = F.leaky_relu(self.conv_layers[i](amino_vector), self.alpha)
        amino_vector = torch.squeeze(amino_vector, 1)
        # print('amino_vector2:',amino_vector.shape)
        amino_vector = F.leaky_relu(self.W_prot(amino_vector), self.alpha)
        return amino_vector

    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a - a_max)
        a_exp = a_exp * mask
        a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
        return a_softmax

    def bidirectional_attention_prediction(self,atoms_vector, atoms_mask, fps, amino_vector, amino_mask, inv_Temp, Temp,epoch,label):
        b = atoms_vector.shape[0]
        # cat_vector_list = []
        for i in range(self.bidat_num):
            A = torch.tanh(torch.matmul(torch.matmul(atoms_vector, self.U[i]), amino_vector.transpose(1, 2)))
            A = A * torch.matmul(atoms_mask.view(b, -1, 1), amino_mask.view(b, 1, -1))

            atoms_trans = torch.matmul(A, torch.tanh(self.transform_p2c[i](amino_vector)))
            amino_trans = torch.matmul(A.transpose(1, 2), torch.tanh(self.transform_c2p[i](atoms_vector)))

            atoms_tmp = torch.cat([torch.tanh(self.bihidden_c[i](atoms_vector)), atoms_trans], dim=2)
            amino_tmp = torch.cat([torch.tanh(self.bihidden_p[i](amino_vector)), amino_trans], dim=2)

            atoms_att = self.mask_softmax(self.biatt_c[i](atoms_tmp).view(b, -1), atoms_mask.view(b, -1))
            amino_att = self.mask_softmax(self.biatt_p[i](amino_tmp).view(b, -1), amino_mask.view(b, -1))

            cf = torch.sum(atoms_vector * atoms_att.view(b, -1, 1), dim=1)
            pf = torch.sum(amino_vector * amino_att.view(b, -1, 1), dim=1)

            if i == 0:
                cat_cf = cf
                cat_pf = pf
            else:
                cat_cf = torch.cat([cat_cf.view(b, -1), cf.view(b, -1)], dim=1)
                cat_pf = torch.cat([cat_pf.view(b, -1), pf.view(b, -1)], dim=1)

        inverse_Temp = inv_Temp.view(inv_Temp.shape[0],-1)
        Temperature = Temp.view(Temp.shape[0],-1)
        cf_final = torch.cat([self.comb_c(cat_cf).view(b, -1), fps.view(b, -1)], dim=1)#length = 2*d
        pf_final = self.comb_p(cat_pf)#length = d
        # print(cf_final.shape,pf_final.shape)
        cat_vector = torch.cat((cf_final, pf_final, inverse_Temp, Temperature), dim=1)#length=3*d+2

        # if label is not None and epoch >= 1:
        #     smoothed_cat_vector = self.Fds.smooth(cat_vector.detach(), label, epoch)
        # else:
        #     smoothed_cat_vector = cat_vector

        for j in range(self.layer_out):
            # smoothed_cat_vector = F.leaky_relu(self.W_out[j](smoothed_cat_vector), self.alpha )
            cat_vector = F.leaky_relu(self.W_out[j](cat_vector), self.alpha )

        # return self.output(smoothed_cat_vector.detach()),smoothed_cat_vector
        # return self.output(cat_vector),smoothed_cat_vector 
        return self.output(cat_vector),cat_vector


    def forward(self, atoms, atoms_mask, adjacency, amino, amino_mask, fps, inv_Temp, Temp ,smiles_names,seq_names,epoch,label):
        
        # print('codeprot:output{cat_vector}')
        atmo_embeddings = []
        for smiles in smiles_names:
            tmp_embedding = self.saved_embedding_dict[smiles]
            # print(tmp_embedding.shape) # n*300*80
            atmo_embeddings.append(tmp_embedding)
            
        # 处理维度
        atmo_embeddings = batch_pad_bert(atmo_embeddings)
        atmo_embeddings = torch.tensor(atmo_embeddings).to('cuda')

        B,N,C = atmo_embeddings.shape
        # print(B,N,C)

        atmo_embeddings = atmo_embeddings.view(-1,C).to(torch.float32)
        atmo_embeddings = self.bert_com_dim(atmo_embeddings)
        atmo_embeddings = atmo_embeddings.view(B,N,self.comp_dim)

        atoms_vector = self.comp_gat(atmo_embeddings, adjacency)

        # atoms_vector = self.comp_gat1(atoms, adjacency)

        # print('atoms:',atoms.shape) 40
        # print('atoms_vector',atoms_vector.shape)

        # amino_vector = self.prot_cnn(amino)
        amino_embeddings = []
        for seq in seq_names:
            tmp_embedding_seq = self.saved_embedding_prot[seq]
            # print(tmp_embedding_seq.shape)
            tmp_embedding_seq_tensor = torch.from_numpy(tmp_embedding_seq).float()
            amino_embeddings.append(tmp_embedding_seq_tensor)
        amino_embeddings_pad,amino_embeddings_mask = batch_pad_seq(amino_embeddings)

        amino_embeddings_pad = amino_embeddings_pad.to('cuda')
        amino_vector = F.leaky_relu(self.prot_com_dim(amino_embeddings_pad), self.alpha)

        super_feature = F.leaky_relu(torch.matmul(fps, self.fp0), 0.1)
        super_feature = F.leaky_relu(torch.matmul(super_feature, self.fp1), 0.1)
        prediction,cat_vector = self.bidirectional_attention_prediction( atoms_vector, atoms_mask, super_feature,\
                                                             amino_vector, amino_mask, inv_Temp, Temp,epoch,label )
        
        
        return prediction,cat_vector