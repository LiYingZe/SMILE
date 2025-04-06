import torch
import torch.nn as nn
from torch.nn import Transformer
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import time
import numpy as np
from tqdm import tqdm
import os
import random
from torch.cuda import amp
import re
import random
import csv

class CharacterLevelDataset(Dataset):
    def __init__(self, src_list,trg_list):
        self.src_list = src_list
        self.trg_list = trg_list
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.build_vocab()

    def build_vocab(self):
        unique_chars = set()
        for sentence in self.src_list + self.trg_list:
            unique_chars.update(list(sentence))
        unique_chars = sorted(unique_chars)
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars, start=3)}
        self.char_to_idx['<SOS>'] = 0
        self.char_to_idx['<EOS>'] = 1
        self.char_to_idx['<PAD>'] = 2
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        self.src_vocab_size = len(self.char_to_idx)
        self.trg_vocab_size = len(self.char_to_idx)

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, index):
        src = self.src_list[index]
        trg = self.trg_list[index]
        
        src = ['<SOS>'] + list(src) + ['<EOS>']
        trg = ['<SOS>'] + list(trg) + ['<EOS>']
        
        src_indices = [self.char_to_idx[char] for char in src]
        trg_indices = [self.char_to_idx[char] for char in trg]
        
        return torch.tensor(src_indices), torch.tensor(trg_indices)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(TransformerSeq2Seq, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, dropout, 5000)
        
        self.output_linear = nn.Linear(d_model, trg_vocab_size)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, trg):
        src_embedding = self.src_embedding(src)
        trg_embedding = self.trg_embedding(trg)
        
        src_embedding = self.positional_encoding(src_embedding)
        trg_embedding = self.positional_encoding(trg_embedding)
        
        src_mask = None
        trg_mask = self.generate_square_subsequent_mask(trg.size(1)).to(trg.device)
        transformer_output = self.transformer(
            src=src_embedding,
            tgt=trg_embedding,
            src_mask=src_mask,
            tgt_mask=trg_mask
        )
        
        output = self.output_linear(transformer_output)
        return output
    


class TransformerClassifier(nn.Module):
    def __init__(self, transformer_seq2seq, num_classes):
        super(TransformerClassifier, self).__init__()
        
        self.encoder = transformer_seq2seq.transformer.encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.src_embedding = transformer_seq2seq.src_embedding
        self.positional_encoding = transformer_seq2seq.positional_encoding
        
        self.classifier = nn.Sequential(
            nn.Linear(transformer_seq2seq.d_model, transformer_seq2seq.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(transformer_seq2seq.d_model // 2, num_classes)
        )
        
    def forward(self, src):
        src = self.src_embedding(src)
        src = self.positional_encoding(src)
        
        encoder_output = self.encoder(src)

        sequence_representation = encoder_output.mean(dim=1)

        logits = self.classifier(sequence_representation)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].permute(1, 0, 2)
        return self.dropout(x)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(iterator, desc="Processing batches")):
        src, trg = batch
        src, trg = src.to(device), trg.to(device)
        
        # Teacher Forcing
        output = model(src, trg[:, :-1])

        output_flattened = output.contiguous().view(-1, output.shape[-1])
        trg_flattened = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_flattened, trg_flattened)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    global device
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch
            src, trg = src.to(device), trg.to(device)
            
            output = model(src, trg[:, :-1])
            
            output_flattened = output.contiguous().view(-1, output.shape[-1])
            trg_flattened = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output_flattened, trg_flattened)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def collate_fn(batch):
    global dataset
    src_batch, trg_batch = [], []
    for src, trg in batch:
        src_batch.append(src)
        trg_batch.append(trg)
    src_padded = pad_sequence(src_batch, padding_value=dataset.char_to_idx['<PAD>'])
    trg_padded = pad_sequence(trg_batch, padding_value=dataset.char_to_idx['<PAD>'])
    return src_padded.T, trg_padded.T


def collate_fn2(batch):
    global dataset
    inputs, outputs = zip(*batch)
    inputs_pad = pad_sequence(inputs,  padding_value=dataset.char_to_idx['<PAD>'])
    outputs = torch.tensor(outputs)
    return inputs_pad.T , outputs 


import json

def read_json_to_dict(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print("The JSON file has been successfully read and converted to a dictionary.")
            return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print("Error: The file content is not in valid JSON format.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def processDict(dic):
    smallQueries = []
    hugeQueries = []
    card2Dict = {}
    for i in range(1000):
        card2Dict[i] = []
    for ki in dic.keys():
        vals = dic[ki]['results']
        if len(vals) ==0 or len(vals) >= 1000:
            continue
        card2Dict[  len(vals) ].append([ki,dic[ki]])

    return card2Dict

import argparse
def print_args(args):

    print("="*40,flush=True)
    print("\nParsed arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("="*40,flush=True)
    print()


def test(x, model, dataset, K=5, temperature=1.0):
    model.eval()
    with torch.no_grad():
        device = x.device
        batch_size = K
        trg_sos = dataset.char_to_idx['<SOS>']
        trg_eos = dataset.char_to_idx['<EOS>']
        
        trg_input = torch.full((batch_size, 1), trg_sos, dtype=torch.long).to(device)
        input_x = x.repeat_interleave(batch_size, dim=0)
        sequence_complete = torch.zeros(batch_size, dtype=torch.bool).to(device)
        generated_sequences = [[] for _ in range(batch_size)]
        
        max_seq_length = 50
        t0 = time.time()
        for _ in range(max_seq_length):
            output = model(input_x, trg_input)
            next_char_probs = output[:, -1, :]
            scaled_probs = next_char_probs / temperature
            next_char_dist = torch.softmax(scaled_probs, dim=1)
            next_char_idx = torch.multinomial(next_char_dist, 1).squeeze(-1)
            next_char_idx[sequence_complete] = trg_eos
            trg_input = torch.cat([trg_input, next_char_idx.unsqueeze(-1)], dim=1)

            new_sequence_complete = next_char_idx == trg_eos
            sequence_complete = sequence_complete | new_sequence_complete

            for i in range(batch_size):
                if not sequence_complete[i]:
                    generated_sequences[i].append(dataset.idx_to_char[next_char_idx[i].item()])
            if all(sequence_complete):
                break
        t1 = time.time()
        print(t1-t0)
        generated_texts = []
        for i in range(batch_size):
            gen_chars = generated_sequences[i]
            generated_texts.append(''.join(gen_chars))
        
        return generated_texts
    


def testBatched(x, model, dataset, K=5, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # import pdb
        # pdb.set_trace()
        device = x.device
        batch_size = K
        trg_sos = dataset.char_to_idx['<SOS>']
        trg_eos = dataset.char_to_idx['<EOS>']
        trg_input = torch.full((batch_size*x.shape[0], 1), trg_sos, dtype=torch.long).to(device)
        input_x = x.repeat_interleave(batch_size, dim=0)
        sequence_complete = torch.zeros(batch_size*x.shape[0], dtype=torch.bool).to(device)
        generated_sequences = [[] for _ in range(batch_size*x.shape[0])]
        max_seq_length = 50

        for _ in range(max_seq_length):

            output = model(input_x, trg_input)
            next_char_probs = output[:, -1, :]
            scaled_probs = next_char_probs / temperature
            next_char_dist = torch.softmax(scaled_probs, dim=1)
            next_char_idx = torch.multinomial(next_char_dist, 1).squeeze(-1)
            next_char_idx[sequence_complete] = trg_eos
            trg_input = torch.cat([trg_input, next_char_idx.unsqueeze(-1)], dim=1)
            new_sequence_complete = next_char_idx == trg_eos
            sequence_complete = sequence_complete | new_sequence_complete
            for i in range(batch_size*x.shape[0]):
                if not sequence_complete[i]:
                    generated_sequences[i].append(dataset.idx_to_char[next_char_idx[i].item()])
            if all(sequence_complete):
                break
        generated_texts = []
        for i in range(batch_size*x.shape[0]):
            gen_chars = generated_sequences[i]
            generated_texts.append(''.join(gen_chars))
        
        return generated_texts
    
def split_list(lst, k):
    num_full_chunks = len(lst) // k
    result = []

    for i in range(num_full_chunks):
        start = i * k
        end = start + k
        result.append(lst[start:end])

    remaining = len(lst) % k
    if remaining > 0:
        result.append(lst[-remaining:])

    return result


class CLSDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return  torch.LongTensor(self.inputs[idx]), self.outputs[idx] 
    
def merge_and_shuffle(L1, L2):
    if len(L1) > len(L2):
        L1, L2 = L2, L1
    
    sub_L2 = L2[:len(L1)]
    
    merged_list = L1 + sub_L2
    
    random.shuffle(merged_list)
    
    return merged_list
def prepare_dataCLS(smallQueries, hugeQueries, batch_size):
    global dataset
    inputs = []
    outputs = []
    sQL = []
    bQL = []
    for sqli,listi in smallQueries:
        sQL.append([sqli,0])
    for sqli,listi in hugeQueries:
        bQL.append([sqli,1])
    
    MergedList = merge_and_shuffle(sQL,bQL)

    for i in range(len(MergedList)):
        s0 = MergedList[i][0]
        s1 = MergedList[i][1]
        inputs.append([ dataset.char_to_idx[c]  for c in s0])
        outputs.append(s1)

    datasetcls = CLSDataset(inputs, outputs)
    dataloader = DataLoader(datasetcls, batch_size=batch_size, shuffle=True, collate_fn=collate_fn2)
    return dataloader
def split_data(X, Y, test_size=0.3, random_seed=42):
    assert len(X) == len(Y), "X and Y must have the same length."

    indices = list(range(len(X)))
    random.seed(random_seed)
    random.shuffle(indices)

    split_index = int(len(indices) * (1 - test_size))

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train = [X[i] for i in train_indices]
    Y_train = [Y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    Y_test = [Y[i] for i in test_indices]

    return X_train, X_test, Y_train, Y_test

def calculate_model_size(model):

    params = model.parameters()
    
    total_elements = sum(p.numel() for p in params)
    
    total_bytes = total_elements * 4

    size_in_mb = total_bytes / (1024 * 1024)
    
    return size_in_mb

def seq2seqTransformer(args ):
    global dataset,device
    tuples = load_data(args.data_path,args.numTem)

    print("Data loaded",flush=True)
    random.shuffle(tuples)
    print("Tuple Shuffled",flush=True)

    X_test = tuples[:256]
    Y_test =  generateListOfWildCards(X_test, insert_char_prob=args.inPct, insert_tail_prob=0.1, wildcard_prob={'%': args.pct , '_': 1-args.pct})
    
    print("Num of training data:",len(tuples))
    
    sqt = []
    for idxx,xs in enumerate(X_test):
        sqt.append([Y_test[idxx], [X_test[idxx]]])
    print(sqt[0])

    print("Chars Stats") 
    dataset=CharacterLevelDataset(X_test+tuples ,Y_test+tuples)   
    print("Num Of Tokens:",len(dataset.char_to_idx.keys()),"Tokens:", len(dataset.char_to_idx.keys())) 
    # import pdb
    # pdb.set_trace()
    print("Chars Stats Done")
    device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
    d_model = args.HIDDEN_SIZE
    nhead = 8
    num_layers = args.LayerNum
    dim_feedforward = args.HIDDEN_SIZE
    max_seq_length = 1024
    dropout = 0.1
    src_vocab_size = dataset.src_vocab_size
    trg_vocab_size = dataset.trg_vocab_size
    model = torch.load(args.PathOfModel,map_location = 'cuda:0').to(device)
    print("Size of Model:",calculate_model_size(model),"MB")
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.char_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs
    clip = 1.0
    Recall = 0
    NumAll =  len(sqt)
    batches = split_list(sqt, args.inferParrellism)
    for inferenceSamples in batches:
        input_texts = []
        labels = []
        queries = []
        for qi in inferenceSamples:
            wildCardi = qi[0]
            results = qi[1]
            input_texts.append(torch.tensor([dataset.char_to_idx['<SOS>']]+[dataset.char_to_idx[c] for c in wildCardi]+[dataset.char_to_idx['<EOS>']]))
            labels.append(results)
            queries.append(wildCardi)
        input_texts = pad_sequence(input_texts, padding_value=dataset.char_to_idx['<PAD>'] ).T.to(device)
        t0 = time.time()
        beamResults =  testBatched(input_texts,model,dataset,K=int(args.inferSampleNum),temperature=1.4)
        t1 = time.time()
        print("Batched Inference Time:",t1-t0, "PerQuery:", (t1-t0)/int(args.inferParrellism),int(args.inferParrellism) )
        predictPerQuery = split_list(beamResults,k=int(args.inferSampleNum))
        for pred_idx in range(len(predictPerQuery)):
            candidates = predictPerQuery[pred_idx]
            results = labels[pred_idx] 
            tRecall = 0
            for vi in results:
                if vi in candidates:
                    tRecall+=1
            Recall+=(tRecall/len(results))
            print("Sample Result:", "wildcards:",queries[pred_idx], "Value:",results, tRecall, len(results) ,  "Result", candidates[:4],len(candidates))
    print(Recall,"Ratio:",Recall/NumAll)

def load_data(data_path,sampleNum = -1):
    if sampleNum==-1:
        print("Using All data as sample")
        import time
        t0 = time.time()
        tuples = []
        with open(data_path, 'r') as f:
            tuples = f.readlines()

        retL = [s.replace("\n", '')[:40] if len(s.replace("\n", '')) > 40 else s.replace("\n", '') for s in tqdm(tuples, desc="Processing lines")]
        t1 = time.time()
        print("Loading Takes:",t1-t0)
    else:
        print(f"Using {sampleNum} data as sample")
        with open(data_path, 'r') as f:
            lines = f.readlines()
        # random_lines = random.sample(lines, sampleNum)
        random_lines = lines[:sampleNum]
        # retL = [s.replace("\n", '') for s in random_lines]
        retL = [s.replace("\n", '')[:40] if len(s.replace("\n", '')) > 40 else s.replace("\n", '') for s in tqdm(random_lines, desc="Processing lines")]
        
    return retL

def select_wildcard(wildcard_prob):
    total = sum(wildcard_prob.values())
    rand = random.uniform(0, total)
    cumulative = 0
    for wc, prob in wildcard_prob.items():
        cumulative += prob
        if rand < cumulative:
            return wc
    return list(wildcard_prob.keys())[-1]

def generate_wildcard_pattern(
    s,
    insert_char_prob=0.5,
    wildcard_prob={'%': 0.5, '_': 0.5},
    insert_tail_prob=0.5,
    percent_skip_prob=0.1
):
    pattern = []
    i = 0
    skipEdNum = 0
    while i < len(s):
        pattern.append(s[i])

        if random.random() < insert_char_prob:
            skipEdNum+=1
            wildcard = select_wildcard(wildcard_prob)
            pattern.append(wildcard)
            if wildcard == '%' and random.random() < percent_skip_prob:
                remaining_chars = len(s) - i - 1
                if remaining_chars > 0:
                    skip = random.randint(0, remaining_chars)
                    i += skip
            else:
                i+=1
        i += 1
    if skipEdNum ==0:
        random_index = random.randint(0, len(pattern) - 1)
        pattern[random_index]="_"
    return '%'+''.join(pattern)+'%'


def generateListOfWildCards(strList,insert_char_prob=0.5, wildcard_prob={'%': 0.5, '_': 0.5} , insert_tail_prob=0.5):
    wildStrings = []
    for si in tqdm(strList, desc="Processing", unit="item"):
        wildStrings.append(generate_wildcard_pattern( si,insert_char_prob, wildcard_prob, insert_tail_prob))
    return wildStrings

def profile_list_of_lists(nested_list):
    length_distribution = {}

    for sublist in nested_list:
        sublist_length = len(sublist[1])
        if sublist_length in length_distribution:
            length_distribution[sublist_length] += 1
        else:
            length_distribution[sublist_length] = 1

    return length_distribution
if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Train a seq2seq model with attention.")
    parser.add_argument('--data_path', type=str, default="./data/lineitem10000.csv", help='Path to data file.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train the model.')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1, help='Teacher forcing ratio for training.')
    parser.add_argument('--max_length', type=int, default=60, help='Maximum length for output sequences.')
    parser.add_argument('--HIDDEN_SIZE', type=int, default=512)
    parser.add_argument('--LayerNum', type=int, default=1)
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--inferSampleNum', type=int, default= 16, help='Number of Inference Samples')
    parser.add_argument('--inferParrellism', type=int, default=64, help='Inference Paralism')
    parser.add_argument('--lr', type=float, default=0.0003, help='Number of samples for inference.')
    parser.add_argument('--seed', type=float, default=42, help='Number of samples for inference.')
    parser.add_argument('--saveName', type=str, default="tmpModel", help='model save folder')
    parser.add_argument('--IterSave', type=int, default= 100, help='Number of Iter Saving Model and Eval')
    parser.add_argument('--pct', type=float, default= 0.2, help='''Ratio of \% in workloads''')
    parser.add_argument('--inPct', type=float,default= 0.2, help='Path Of Evaling Model')
    parser.add_argument('--numTem', type=int, default= -1, help='Number of temtemplate')
    parser.add_argument('--numRepeat', type=int, default= 1, help='Number of temtemplate seedOrigin')
    parser.add_argument('--PathOfModel', type=str, default="/home/lyz/LLM_PrefixSearch/models/lineitem10000_lr0.0003_in1Pct2/Ep_9999_Seq2Seq", help='Path Of Evaling Model')
    parser.add_argument('--QueryPath', type=str, default= "/home/lyz/LLM_PrefixSearch/TrainingWorkloads/TPCH4/lineitem10000_ipct1.0_pct2.0_new.json", help='Number of temtemplate seedOrigin')
    args = parser.parse_args()
    print_args(args)
    seq2seqTransformer(args ) 