from Mole_BERT.loader import mol_to_graph_data_obj_simple
from Mole_BERT.model import GNN_graphpred
from transformers import T5Tokenizer, T5EncoderModel
from rdkit import Chem
import pandas as pd
import torch
import pickle
from tqdm import tqdm
import argparse
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inputs: --data_path: path to data file; \
                                    --output_path_bert: path to output dir for BERT results; \
                                    --output_path_prot: path to output dir for Protein results')
    parser.add_argument('--data_path', type=str, default='/home/supermicro/code/AI4Sci/DLTKcat/data/pkl/prot_bert_dlt_mnk_sjh_mpek/all_data.csv')
    parser.add_argument('--output_path_bert', type=str, default='/home/supermicro/code/AI4Sci/DLTKcat/data/pred/bert/bert_dict.pkl')
    parser.add_argument('--output_path_prot', type=str, default='/home/supermicro/code/AI4Sci/DLTKcat/data/pred/prot/prot_dict.pkl')
    args = parser.parse_args()

    data_path = str(args.data_path)
    output_path_bert = str(args.output_path_bert)
    output_path_prot = str(args.output_path_prot)

    df = pd.read_csv(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prot_model = '/home/supermicro/code/AI4Sci/DLTKcat/prot_t5_xl_uniref50'
    bert_model = '/home/supermicro/code/AI4Sci/DLTKcat/Mole_BERT/model_gin/Mole-BERT.pth' 

    bert_encoder = GNN_graphpred(5, 300, num_tasks=1, drop_ratio=0.5)
    if bert_model is not None and bert_model != "":
        bert_encoder.from_pretrained(bert_model)
        print("Loaded pre-trained model with success.")
    else:
        raise FileNotFoundError("bert model path not found")

    if prot_model:
        tokenizer = T5Tokenizer.from_pretrained(prot_model,do_lower_case=False)
        # print(f"最大输入长度: {tokenizer.model_max_length}")
        prot_encoder = T5EncoderModel.from_pretrained(prot_model).to(device)
        print("Loaded pre-trained Protein model successfully.")
    else:
        raise FileNotFoundError("prot model path not found")

    bert_results_dict = {}
    prot_results_dict = {}

    simels_list = df['smiles'].tolist()
    seq_list = df['seq'].tolist()
    
    bert_encoder.eval()
    for smiles in tqdm(simels_list,desc="Processing"):

        data = mol_to_graph_data_obj_simple(Chem.AddHs(Chem.MolFromSmiles(smiles)))
        with torch.no_grad():
            graph_pred, node_representation = bert_encoder(data)
            bert_results_dict[smiles] = node_representation

    with open(output_path_bert, 'wb') as f:
        pickle.dump(bert_results_dict, f)
    print('success get pred data bert pkl')

    prot_encoder.eval()
    for seq in tqdm(seq_list, desc="Processing Sequences"):
        seq_length = len(seq)
        seq_re = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]
        ids = tokenizer.batch_encode_plus(seq_re, add_special_tokens=True, padding='max_length', truncation=True, max_length=seq_length)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            outputs = prot_encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_seq = outputs.last_hidden_state
            seq_representation = outputs.last_hidden_state.squeeze().cpu().numpy()
            # print(seq_length,seq_representation.shape)
            prot_results_dict[seq] = seq_representation

    with open(output_path_prot, 'wb') as f:
        pickle.dump(prot_results_dict, f)
    print('Success: Protein results saved to', output_path_prot)
