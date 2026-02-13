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
import pyphen
from torch.cuda import amp
import re
import csv
import json
import argparse
from torch.optim import AdamW
from typing import List, Tuple, Dict, Optional, Union
import warnings


class ImprovedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(ImprovedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            warnings.warn(f"Sequence length {seq_len} exceeds max_len {self.pe.size(0)}. "
                         "Consider increasing max_len or truncating sequences.")
            seq_len = self.pe.size(0)
            x = x[:, :seq_len]
    
        x = x + self.pe[:seq_len, :].permute(1, 0, 2)
        return self.dropout(x)


class OptimizedTransformerSeq2Seq(nn.Module):
    def __init__(self, 
                 src_vocab_size: int, 
                 trg_vocab_size: int, 
                 d_model: int, 
                 nhead: int, 
                 num_layers: int, 
                 dim_feedforward: int, 
                 max_seq_length: int, 
                 dropout: float = 0.1):
        super(OptimizedTransformerSeq2Seq, self).__init__()
        self.d_model = d_model
        
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
        self.positional_encoding = ImprovedPositionalEncoding(d_model, dropout, max_seq_length)
        
        self.output_linear = nn.Linear(d_model, trg_vocab_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for embedding in [self.src_embedding, self.trg_embedding]:
            nn.init.normal_(embedding.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.constant_(self.output_linear.bias, 0)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src: torch.Tensor, trg: torch.Tensor, 
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_embedding = self.src_embedding(src) * math.sqrt(self.d_model)
        trg_embedding = self.trg_embedding(trg) * math.sqrt(self.d_model)

        src_embedding = self.positional_encoding(src_embedding)
        trg_embedding = self.positional_encoding(trg_embedding)

        trg_mask = self.generate_square_subsequent_mask(trg.size(1)).to(trg.device)
        src_mask = None

        transformer_output = self.transformer(
            src=src_embedding,
            tgt=trg_embedding,
            src_mask=src_mask,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        output = self.output_linear(transformer_output)
        return output


class OptimizedTransformerClassifier(nn.Module):
    def __init__(self, transformer_seq2seq: OptimizedTransformerSeq2Seq, num_classes: int):
        super(OptimizedTransformerClassifier, self).__init__()
        self.d_model = transformer_seq2seq.d_model

        self.encoder = transformer_seq2seq.transformer.encoder

        self.src_embedding = transformer_seq2seq.src_embedding
        self.positional_encoding = transformer_seq2seq.positional_encoding

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, num_classes)
        )
        
    def forward(self, src: torch.Tensor, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        
        encoder_output = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.unsqueeze(-1)
            encoder_output = encoder_output.masked_fill(mask, 0.0)
            seq_lengths = (~mask).sum(dim=1)
            sequence_representation = encoder_output.sum(dim=1) / seq_lengths.clamp(min=1)
        else:
            sequence_representation = encoder_output.mean(dim=1)
        
        logits = self.classifier(sequence_representation)
        return logits


def optimized_train(model: nn.Module, iterator: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, clip: float, device: torch.device, 
                   accumulation_steps: int = 1, use_amp: bool = False) -> float:
    model.train()
    epoch_loss = 0
    scaler = amp.GradScaler(enabled=use_amp)
    
    for i, batch in enumerate(tqdm(iterator, desc="Training Batches")):
        src, trg = batch
        src, trg = src.to(device), trg.to(device)

        src_padding_mask = (src == 2)
        trg_padding_mask = (trg == 2)
        
        with amp.autocast(enabled=use_amp):
            output = model(src, trg[:, :-1], src_padding_mask, trg_padding_mask[:, :-1])
            
            output_flattened = output.contiguous().view(-1, output.shape[-1])
            trg_flattened = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_flattened, trg_flattened)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps

    return epoch_loss / len(iterator)

def optimized_evaluate(model: nn.Module, iterator: DataLoader, criterion: nn.Module, device: torch.Tensor) -> float:
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            src, trg = batch
            src, trg = src.to(device), trg.to(device)
            
            src_padding_mask = (src == 2)
            trg_padding_mask = (trg == 2)

            output = model(src, trg[:, :-1], src_padding_mask, trg_padding_mask[:, :-1])
            
            output_flattened = output.contiguous().view(-1, output.shape[-1])
            trg_flattened = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_flattened, trg_flattened)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def optimized_test(x: torch.Tensor, model: nn.Module, dataset: Dataset, 
                  device: torch.device, K: int = 5, temperature: float = 1.0) -> List[str]:
    model.eval()
    with torch.no_grad():
        batch_size = K
        trg_sos = dataset.char_to_idx['<SOS>']
        trg_eos = dataset.char_to_idx['<EOS>']
        
        input_x = x.repeat_interleave(batch_size, dim=0).to(device)
        
        trg_input = torch.full((batch_size * x.shape[0], 1), trg_sos, dtype=torch.long, device=device)
        
        sequence_complete = torch.zeros(batch_size * x.shape[0], dtype=torch.bool, device=device)
        generated_sequences = [[] for _ in range(batch_size * x.shape[0])]
        
        max_seq_length = 50
        
        for _ in range(max_seq_length):
            trg_padding_mask = (trg_input == 2)
            
            output = model(input_x, trg_input, tgt_key_padding_mask=trg_padding_mask)
            
            next_char_probs = output[:, -1, :] 

            scaled_probs = next_char_probs / temperature
            next_char_dist = torch.softmax(scaled_probs, dim=1)

            next_char_idx = torch.multinomial(next_char_dist, 1).squeeze(-1)

            next_char_idx[sequence_complete] = trg_eos
            trg_input = torch.cat([trg_input, next_char_idx.unsqueeze(-1)], dim=1)

            new_sequence_complete = (next_char_idx == trg_eos)
            sequence_complete = sequence_complete | new_sequence_complete

            for i in range(batch_size * x.shape[0]):
                if not sequence_complete[i]:
                    generated_sequences[i].append(dataset.idx_to_char[next_char_idx[i].item()])

            if all(sequence_complete):
                break

        generated_texts = [''.join(seq) for seq in generated_sequences]
        return generated_texts

def setup_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("CUDA not available or selected, using CPU.")
    return device

class CharacterLevelDataset(Dataset):
    def __init__(self, src_list: List[str], trg_list: List[str]):
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

        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.src_vocab_size = len(self.char_to_idx)
        self.trg_vocab_size = len(self.char_to_idx)

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.src_list[index]
        trg = self.trg_list[index]
        
        src_chars = ['<SOS>'] + list(src) + ['<EOS>']
        trg_chars = ['<SOS>'] + list(trg) + ['<EOS>']
        
        src_indices = [self.char_to_idx[char] for char in src_chars]
        trg_indices = [self.char_to_idx[char] for char in trg_chars]
        
        return torch.tensor(src_indices), torch.tensor(trg_indices)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    src_batch, trg_batch = [], []
    for src, trg in batch:
        src_batch.append(src)
        trg_batch.append(trg)
    padding_value = dataset.char_to_idx['<PAD>']
    src_padded = pad_sequence(src_batch, padding_value=padding_value, batch_first=True)
    trg_padded = pad_sequence(trg_batch, padding_value=padding_value, batch_first=True)
    return src_padded, trg_padded

def collate_fn2(batch):
    inputs, outputs = zip(*batch)
    padding_value = dataset.char_to_idx['<PAD>']
    inputs_pad = pad_sequence(inputs, padding_value=padding_value, batch_first=True)
    outputs_tensor = torch.tensor(outputs, dtype=torch.long)
    return inputs_pad, outputs_tensor 

def read_json_to_dict(file_path: str) -> Optional[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: file {file_path} not found.")
    except json.JSONDecodeError:
        print("Error: invalid JSON content.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None

def processDict(dic: Dict, minimalLength: int = 10):
    smallQueries = []
    hugeQueries = []
    for ki in dic.keys():
        if len(dic[ki]) == 0: continue
        if len(dic[ki]) <= minimalLength:
            smallQueries.append([ki, dic[ki]])
        else:
            hugeQueries.append([ki, dic[ki]])
    print("Data processed. MeanLen of Small:", np.mean([len(vi[1]) for vi in smallQueries]) if smallQueries else 0, 
          ". MeanLen of Huge:", np.mean([len(vi[1]) for vi in hugeQueries]) if hugeQueries else 0)
    print("Data processed. Number of Small:", len(smallQueries), 
          ". Number of Huge:", len(hugeQueries))
    return smallQueries, hugeQueries

def print_args(args: argparse.Namespace):
    print("="*40, flush=True)
    print("\nParsed arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("="*40, flush=True)

def calculate_model_size(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def split_list(lst: List, k: int) -> List[List]:
    num_full_chunks = len(lst) // k
    result = [lst[i * k : (i + 1) * k] for i in range(num_full_chunks)]
    remaining = len(lst) % k
    if remaining > 0:
        result.append(lst[-remaining:])
    return result

class CLSDataset(Dataset):
    def __init__(self, inputs: List[List[int]], outputs: List[int]):
        self.inputs = inputs
        self.outputs = outputs
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx: int):
        return torch.LongTensor(self.inputs[idx]), self.outputs[idx] 

def merge_and_shuffle(L1: List, L2: List) -> List:
    if len(L1) > len(L2):
        L1, L2 = L2, L1
    sub_L2 = L2[:len(L1)]
    merged_list = L1 + sub_L2
    random.shuffle(merged_list)
    return merged_list

def prepare_dataCLS(smallQueries: List, hugeQueries: List, batch_size: int) -> DataLoader:
    global dataset
    inputs = []
    outputs = []
    sQL, bQL = [], []
    for sqli, _ in smallQueries: sQL.append([sqli, 0])
    for sqli, _ in hugeQueries: bQL.append([sqli, 1])
    
    MergedList = merge_and_shuffle(sQL, bQL)

    for s0, s1 in MergedList:
        inputs.append([dataset.char_to_idx[c] for c in s0])
        outputs.append(s1)
        
    datasetcls = CLSDataset(inputs, outputs)
    dataloader = DataLoader(datasetcls, batch_size=batch_size, shuffle=True, collate_fn=collate_fn2)
    return dataloader

def split_data(X: List, Y: List, test_size: float = 0.3, random_seed: int = 42):
    assert len(X) == len(Y)
    indices = list(range(len(X)))
    random.seed(random_seed)
    random.shuffle(indices)
    split_idx = int(len(indices) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    Y_train = [Y[i] for i in train_idx]
    Y_test = [Y[i] for i in test_idx]
    return X_train, X_test, Y_train, Y_test

def select_wildcard(wildcard_prob: Dict[str, float]) -> str:
    total = sum(wildcard_prob.values())
    rand = random.uniform(0, total)
    cumulative = 0
    for wc, prob in wildcard_prob.items():
        cumulative += prob
        if rand < cumulative:
            return wc
    return list(wildcard_prob.keys())[-1]

def generate_wildcard_pattern(
    s: str,
    insert_char_prob: float = 0.5,
    wildcard_prob: Dict[str, float] = {'%': 0.5, '_': 0.5},
    insert_tail_prob: float = 0.5,
    percent_skip_prob: float = 0.1
) -> str:
    if len(s) < 5:
        return f"%{s}%"
    else:
        return _original_strategy(s, wildcard_prob, insert_tail_prob, percent_skip_prob,insert_char_prob)


def _original_strategy(
    s: str,
    wildcard_prob: Dict[str, float],
    insert_tail_prob: float,
    percent_skip_prob: float,
    insert_char_prob: float
) -> str:
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

def generateListOfWildCards(str_list: List[str], insert_char_prob: float = 0.5, 
                          wildcard_prob: Dict[str, float] = {'%': 0.5, '_': 0.5}) -> List[str]:
    wild_strings = []
    for s in tqdm(str_list, desc="Generating Wildcards", unit="item"):
        wild_strings.append(generate_wildcard_pattern(s, insert_char_prob, wildcard_prob))
    return wild_strings

def load_data(data_path: str, sample_num: int = -1) -> List[str]:
    import string
    valid_chars = set(string.ascii_letters + string.digits + string.punctuation + ' ')
    lines = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading data file {data_path}: {e}")
        return []

    if sample_num != -1:
        lines = lines[:sample_num]
        
    ret_l = []
    for s in tqdm(lines, desc="Processing lines"):
        s_clean = s.replace("\n", '')
        if all(c in valid_chars for c in s_clean):
            s_truncated = s_clean[:40] if len(s_clean) > 40 else s_clean
            ret_l.append(s_truncated.replace('%', '_').replace('_','%')) 
    return ret_l

def profile_list_of_lists(nested_list: List[List]):
    length_distribution = {}
    for sublist in nested_list:
        sublist_length = len(sublist[1])
        if sublist_length in length_distribution:
            length_distribution[sublist_length] += 1
        else:
            length_distribution[sublist_length] = 1
    return length_distribution

def setup_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seq2seqTransformer(args, smallQueries, hugeQueries):
    global dataset, device

    device = setup_device(args.GPU)
    print(f"Using device: {device}")
    
    tuples = load_data(args.data_path, args.numTem)
    if not tuples:
        print("Failed to load data. Exiting.")
        return
        
    lengths = [len(t) for t in tuples]
    print(f"Data stats - avg={np.mean(lengths):.2f}, min={min(lengths)}, max={max(lengths)}")
    
    X_test = tuples[:256]
    Y_test = generateListOfWildCards(X_test, args.inPct, wildcard_prob={'%': args.pct, '_': 1-args.pct})
    
    all_src = X_test + tuples
    all_trg = Y_test
    dataset = CharacterLevelDataset(all_src, all_trg)
    print(f"Num Of Tokens: {len(dataset.char_to_idx)}")

    model_params = {
        'src_vocab_size': dataset.src_vocab_size,
        'trg_vocab_size': dataset.trg_vocab_size,
        'd_model': args.HIDDEN_SIZE,
        'nhead': 8,
        'num_layers': args.LayerNum,
        'dim_feedforward': args.HIDDEN_SIZE,
        'max_seq_length': 1024,
        'dropout': 0.1
    }
    model = OptimizedTransformerSeq2Seq(**model_params).to(device)
    print(f"Model size: {calculate_model_size(model):.2f} MB")

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.char_to_idx['<PAD>'])
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    clip = 1.0
    model_save_root = f'./models/{args.saveName}'
    os.makedirs(model_save_root, exist_ok=True)
    
    use_amp_enabled = device.type == 'cuda'
    grad_accum_steps = 4

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}", flush=True)

        train_sample_indices = random.sample(range(len(tuples)),  len(tuples) )
        X_train_sample = [tuples[i] for i in train_sample_indices]
        Y_train_sample = generateListOfWildCards(X_train_sample, args.inPct, wildcard_prob={'%': args.pct, '_': 1 - args.pct})
        
        dataset_train = CharacterLevelDataset(Y_train_sample, X_train_sample)
        train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        start_time = time.time()

        train_loss = optimized_train(model, train_dataloader, optimizer, criterion, clip, device, 
                                    accumulation_steps=grad_accum_steps, use_amp=use_amp_enabled)
        
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Time: {end_time-start_time:.2f}s")

        if epoch % args.IterSave == (args.IterSave - 1):
            print("Saving model...")
            torch.save(model.state_dict(), os.path.join(model_save_root, f'Ep_X_Seq2Seq_state_dict.pth'))

            print("Evaluating Recall...")
            sqt = [[Y_test[i], [X_test[i]]] for i in range(len(X_test))]
            recall = 0
            num_all = len(sqt)
            batches = split_list(sqt, args.inferParrellism)
            
            for inference_samples in batches:
                input_texts = []
                labels = []
                queries = []
                for wi, res_list in inference_samples:
                    queries.append(wi)
                    labels.append(res_list)
                    input_indices = [dataset.char_to_idx['<SOS>']] + [dataset.char_to_idx[c] for c in wi] + [dataset.char_to_idx['<EOS>']]
                    input_texts.append(torch.tensor(input_indices))
                
                input_texts_padded = pad_sequence(input_texts, padding_value=dataset.char_to_idx['<PAD>'], batch_first=True).to(device)
                
                t0 = time.time()
                beam_results = optimized_test(input_texts_padded, model, dataset, device, K=int(args.inferSampleNum), temperature=0.9)
                t1 = time.time()
                print(f"Batched Inference Time: {t1-t0:.2f}s, Per Query: {(t1-t0)/len(inference_samples):.4f}s")
                
                predict_per_query = split_list(beam_results, k=int(args.inferSampleNum))
                
                for pred_idx, (candidates, ground_truth) in enumerate(zip(predict_per_query, labels)):
                    hit_count = sum(1 for gt in ground_truth if gt in candidates)
                    recall += (hit_count / len(ground_truth)) if ground_truth else 0
                    if pred_idx < 2: 
                         print(f"Sample Result | Wildcard: {queries[pred_idx]}, Ground Truth: {ground_truth}, Recall: {hit_count}/{len(ground_truth)}, Predictions: {candidates[:3]}...")

            print(f"Total Recall: {recall:.4f}, Ratio: {recall/num_all:.4f}")

    print("Seq2Seq Model training completed.")
    return
    print("Starting Classifier Training...")
    nn_cls = OptimizedTransformerClassifier(model, num_classes=2).to(device)
    for param in nn_cls.encoder.parameters():
        param.requires_grad = False

    cls_criterion = nn.CrossEntropyLoss()
    cls_optimizer = AdamW(nn_cls.parameters(), lr=args.lr * 0.1)

    if not smallQueries or not hugeQueries:
        print("Skipping classifier training due to empty data.")
        return

    cls_dataloader = prepare_dataCLS(smallQueries, hugeQueries, args.batch_size)
    
    for epoch in range(args.epochs):
        nn_cls.train()
        total_loss = 0
        for batch_idx, (input_batch, labels_batch) in enumerate(tqdm(cls_dataloader, desc="Classifier Training")):
            input_batch, labels_batch = input_batch.to(device), labels_batch.to(device)
            cls_optimizer.zero_grad()
            
            src_padding_mask = (input_batch == dataset.char_to_idx['<PAD>'])
            outputs = nn_cls(input_batch, src_padding_mask)
            
            loss = cls_criterion(outputs, labels_batch)
            loss.backward()
            cls_optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Classifier Loss: {total_loss / len(cls_dataloader):.4f}")
        
        nn_cls.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for input_batch, labels_batch in cls_dataloader:
                input_batch, labels_batch = input_batch.to(device), labels_batch.to(device)
                src_padding_mask = (input_batch == dataset.char_to_idx['<PAD>'])
                outputs = nn_cls(input_batch, src_padding_mask)
                _, predicted = torch.max(outputs, dim=1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}], Classifier Accuracy: {accuracy:.2f}%")

        if epoch % args.IterSave == (args.IterSave - 1):
            print("Saving classifier model...")
            torch.save(nn_cls.state_dict(), os.path.join(model_save_root, f'Ep_{epoch}_CLS_state_dict.pth'))

    print("All training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a seq2seq model with optimized Transformer.")
    parser.add_argument('--data_path', type=str, default="./data/lineitem.csv", help='Path to data file.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train the model.')
    parser.add_argument('--HIDDEN_SIZE', type=int, default=512)
    parser.add_argument('--LayerNum', type=int, default=4)
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--inferSampleNum', type=int, default=4, help='Number of Inference Samples')
    parser.add_argument('--inferParrellism', type=int, default=64, help='Inference Paralism')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--saveName', type=str, default="tmpModel", help='model save folder')
    parser.add_argument('--IterSave', type=int, default=50, help='Number of Iter Saving Model and Eval')
    parser.add_argument('--pct', type=float, default=0.2, help='''Ratio of \\% in workloads''')
    parser.add_argument('--inPct', type=float, default=0.1, help='workload type for W3 W4')
    parser.add_argument('--numTem', type=int, default=-1, help='Number of templates to use')
    parser.add_argument('--numRepeat', type=int, default=1, help='Number of template seeds')
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)
    print_args(args)
    
    smallQueries, hugeQueries = [], []

    seq2seqTransformer(args, smallQueries, hugeQueries)
