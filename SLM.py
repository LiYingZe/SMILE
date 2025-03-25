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
import csv
import json
import argparse

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
        # 添加特殊标记
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
        
        # Transformer 模型
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout, 5000)
        
        # 输出层
        self.output_linear = nn.Linear(d_model, trg_vocab_size)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, trg):
        # 源序列和目标序列的嵌入
        src_embedding = self.src_embedding(src)
        trg_embedding = self.trg_embedding(trg)
        
        # 添加位置编码
        src_embedding = self.positional_encoding(src_embedding)
        trg_embedding = self.positional_encoding(trg_embedding)
        
        # Transformer 的掩码
        src_mask = None
        trg_mask = self.generate_square_subsequent_mask(trg.size(1)).to(trg.device)
        
        # Transformer 的输出
        transformer_output = self.transformer(
            src=src_embedding,
            tgt=trg_embedding,
            src_mask=src_mask,
            tgt_mask=trg_mask
        )
        
        # 输出层
        output = self.output_linear(transformer_output)
        return output
    


class TransformerClassifier(nn.Module):
    def __init__(self, transformer_seq2seq, num_classes):
        super(TransformerClassifier, self).__init__()
        
        # 提取并冻结编码器
        self.encoder = transformer_seq2seq.transformer.encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        # 提取嵌入层和位置编码
        self.src_embedding = transformer_seq2seq.src_embedding
        self.positional_encoding = transformer_seq2seq.positional_encoding
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(transformer_seq2seq.d_model, transformer_seq2seq.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(transformer_seq2seq.d_model // 2, num_classes)
        )
        
    def forward(self, src):
        # 嵌入和位置编码
        src = self.src_embedding(src)
        src = self.positional_encoding(src)
        
        # 编码器输出
        encoder_output = self.encoder(src)
        
        # 取编码器输出的平均值作为序列的表示
        sequence_representation = encoder_output.mean(dim=1)
        
        # 分类器
        logits = self.classifier(sequence_representation)
        return logits

# class TransformerSeq2Seq(nn.Module):
#     def __init__(self, src_vocab_size, trg_vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout=0.1):
#         super(TransformerSeq2Seq, self).__init__()
#         self.src_vocab_size = src_vocab_size
#         self.trg_vocab_size = trg_vocab_size
#         self.d_model = d_model
#         self.nhead = nhead
#         self.num_layers = num_layers
#         self.dim_feedforward = dim_feedforward
#         self.max_seq_length = max_seq_length
#         self.dropout = dropout
        
#         # Transformer 模型
#         self.transformer = Transformer(
#             d_model=d_model,
#             nhead=nhead,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=num_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         # 嵌入层
#         self.src_embedding = nn.Embedding(src_vocab_size, d_model)
#         self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
#         # 位置编码
#         self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
#         # 输出层
#         self.output_linear = nn.Linear(d_model, trg_vocab_size)
    
#     def generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask
    
#     def forward(self, src, trg):
#         # 源序列和目标序列的嵌入
#         src_embedding = self.src_embedding(src)
#         trg_embedding = self.trg_embedding(trg)
        
#         # 添加位置编码
#         src_embedding = self.positional_encoding(src_embedding)
#         trg_embedding = self.positional_encoding(trg_embedding)
        
#         # Transformer 的掩码
#         src_mask = None
#         trg_mask = self.generate_square_subsequent_mask(trg.size(1)).to(trg.device)
        
#         # Transformer 的输出
#         transformer_output = self.transformer(
#             src=src_embedding,
#             tgt=trg_embedding,
#             src_mask=src_mask,
#             tgt_mask=trg_mask
#         )

#         # 输出层
#         output = self.output_linear(transformer_output)
#         return output
    

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
 
        # pdb.set_trace()
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
        
        # 将目标序列展开为一维张量
        output_flattened = output.contiguous().view(-1, output.shape[-1])
        trg_flattened = trg[:, 1:].contiguous().view(-1)

        # 计算损失
        loss = criterion(output_flattened, trg_flattened)
        loss.backward()
        
        # 梯度裁剪
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


# 创建 DataLoader
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



def read_json_to_dict(file_path):
    """
    读取JSON文件并将其内容转换为字典。
    
    参数:
        file_path (str): JSON文件的路径。
    
    返回:
        dict: JSON文件内容转换后的字典。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print("JSON文件已成功读取并转换为字典。")
            return data
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
        return None
    except json.JSONDecodeError:
        print("错误：文件内容不是有效的JSON格式。")
        return None
    except Exception as e:
        print(f"发生错误：{e}")
        return None


def processDict(dic, minimalLength=10):
    smallQueries = []
    hugeQueries = []
    for ki in dic.keys():
        if len(dic[ki]) ==0:
            continue
        if len(dic[ki]) <= minimalLength:
            smallQueries.append([ki,dic[ki]])
        else:
            hugeQueries.append([ki,dic[ki]])
    print("data processed. MeanLen of Small:", np.mean([ len(vi[1]) for vi in smallQueries ]) ,". MeanLen of Huge:", np.mean([ len(vi[1]) for vi in hugeQueries ]) )
    print("data processed. Number of Small:", len(smallQueries) ,". MeanLen of Huge:", len(hugeQueries) )

    return smallQueries,hugeQueries
# 测试输入
def print_args(args):
    """
    打印argparse解析的所有参数。
    :param args: argparse.Namespace对象，包含解析后的参数。
    """
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
        
        # 初始化目标输入，每个样本以 <SOS> 开头
        trg_input = torch.full((batch_size, 1), trg_sos, dtype=torch.long).to(device)
        # 扩展输入 x 的批次大小为 K
        input_x = x.repeat_interleave(batch_size, dim=0)
        sequence_complete = torch.zeros(batch_size, dtype=torch.bool).to(device)
        generated_sequences = [[] for _ in range(batch_size)]
        
        max_seq_length = 50  # 最大生成长度
        t0 = time.time()
        # 并行生成所有 K 个序列
        for _ in range(max_seq_length):
            # 计算模型输出
            output = model(input_x, trg_input)
            next_char_probs = output[:, -1, :]  # 取最后一个时间步的输出概率
            # 应用温度采样
            scaled_probs = next_char_probs / temperature
            next_char_dist = torch.softmax(scaled_probs, dim=1)
            # 按概率采样下一个字符
            next_char_idx = torch.multinomial(next_char_dist, 1).squeeze(-1)
            # 更新目标序列
            next_char_idx[sequence_complete] = trg_eos  # 已完成的序列保持 <EOS>
            trg_input = torch.cat([trg_input, next_char_idx.unsqueeze(-1)], dim=1)
            # 检查是否生成 <EOS>
            new_sequence_complete = next_char_idx == trg_eos
            sequence_complete = sequence_complete | new_sequence_complete
            # 收集生成字符
            for i in range(batch_size):
                if not sequence_complete[i]:
                    generated_sequences[i].append(dataset.idx_to_char[next_char_idx[i].item()])
            # 如果所有序列都已完成，提前退出
            if all(sequence_complete):
                break
        t1 = time.time()
        print(t1-t0)
        # 将生成的序列转换为字符串
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
        # 初始化目标输入，每个样本以 <SOS> 开头
        trg_input = torch.full((batch_size*x.shape[0], 1), trg_sos, dtype=torch.long).to(device)
        # 扩展输入 x 的批次大小为 K
        input_x = x.repeat_interleave(batch_size, dim=0)
        sequence_complete = torch.zeros(batch_size*x.shape[0], dtype=torch.bool).to(device)
        generated_sequences = [[] for _ in range(batch_size*x.shape[0])]
        max_seq_length = 50  # 最大生成长度
        
        # 并行生成所有 K 个序列
        for _ in range(max_seq_length):
            # 计算模型输出
            output = model(input_x, trg_input)
            next_char_probs = output[:, -1, :]  # 取最后一个时间步的输出概率
            # 应用温度采样
            scaled_probs = next_char_probs / temperature
            next_char_dist = torch.softmax(scaled_probs, dim=1)
            # 按概率采样下一个字符
            next_char_idx = torch.multinomial(next_char_dist, 1).squeeze(-1)
            # 更新目标序列
            next_char_idx[sequence_complete] = trg_eos  # 已完成的序列保持 <EOS>
            trg_input = torch.cat([trg_input, next_char_idx.unsqueeze(-1)], dim=1)
            # 检查是否生成 <EOS>
            new_sequence_complete = next_char_idx == trg_eos
            sequence_complete = sequence_complete | new_sequence_complete
            # 收集生成字符
            for i in range(batch_size*x.shape[0]):
                if not sequence_complete[i]:
                    generated_sequences[i].append(dataset.idx_to_char[next_char_idx[i].item()])
            # 如果所有序列都已完成，提前退出
            if all(sequence_complete):
                break
        # 将生成的序列转换为字符串
        generated_texts = []
        for i in range(batch_size*x.shape[0]):
            gen_chars = generated_sequences[i]
            generated_texts.append(''.join(gen_chars))
        
        return generated_texts
    
def split_list(lst, k):
    # 计算可以完整划分的子列表数量
    num_full_chunks = len(lst) // k
    # 初始化结果列表
    result = []

    # 处理完整划分的部分
    for i in range(num_full_chunks):
        start = i * k
        end = start + k
        result.append(lst[start:end])

    # 处理剩余的元素
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
    # 确保 L1 是较短的列表，L2 是较长的列表
    if len(L1) > len(L2):
        L1, L2 = L2, L1
    
    # 从 L2 中取出长度为 L1 的子列表
    sub_L2 = L2[:len(L1)]
    
    # 合并 L1 和 sub_L2
    merged_list = L1 + sub_L2
    
    # 打散合并后的列表
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
    # import pdb
    # pdb.set_trace()

    datasetcls = CLSDataset(inputs, outputs)
    dataloader = DataLoader(datasetcls, batch_size=batch_size, shuffle=True, collate_fn=collate_fn2)
    return dataloader
def split_data(X, Y, test_size=0.3, random_seed=42):
    # 确保输入的X和Y长度相同
    assert len(X) == len(Y), "X and Y must have the same length."

    # 复制索引列表
    indices = list(range(len(X)))
    random.seed(random_seed)  # 设置随机种子，确保结果可复现
    random.shuffle(indices)  # 打乱索引顺序

    # 计算测试集的大小
    split_index = int(len(indices) * (1 - test_size))

    # 分割索引
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    # 根据索引分割数据
    X_train = [X[i] for i in train_indices]
    Y_train = [Y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    Y_test = [Y[i] for i in test_indices]

    return X_train, X_test, Y_train, Y_test

def calculate_model_size(model):
    """
    计算给定 PyTorch 模型的参数规模（以 MB 为单位）。

    参数:
        model (torch.nn.Module): 需要计算的神经网络模型。

    返回:
        float: 模型参数大小（MB）。
    """
    # 获取模型的参数
    params = model.parameters()
    
    # 计算参数的总元素数量
    total_elements = sum(p.numel() for p in params)
    
    # 计算以 Byte 为单位的大小（假设每个参数是 float32 类型，占用 4 字节）
    total_bytes = total_elements * 4
    
    # 转换为 MB（1MB = 1024 * 1024 字节）
    size_in_mb = total_bytes / (1024 * 1024)
    
    return size_in_mb

def seq2seqTransformer(args,smallQueries,hugeQueries ):
    global dataset,device
    tuples = load_data(args.data_path,args.numTem)
    print("Data loaded",flush=True)
    # random.shuffle(tuples)
    print("Tuple Shuffled",flush=True)
    X_test = tuples[:256]
    Y_test =  generateListOfWildCards(X_test, args.inPct, insert_tail_prob=0.1, wildcard_prob={'%': args.pct , '_': 1-args.pct})
    print("Num of training data:",len(tuples))
    sqt = []
    for idxx,xs in enumerate(X_test):
        sqt.append([Y_test[idxx], [X_test[idxx]]])
    # import pdb
    # pdb.set_trace()
    modelSaveRoot = f'./models/{args.saveName}'
    os.makedirs(modelSaveRoot, exist_ok=True)
    
    print("Chars Stats") 
    dataset=CharacterLevelDataset(X_test+tuples ,Y_test+tuples)   
    print("Num Of Tokens:",len(dataset.char_to_idx.keys()),"Tokens:", len(dataset.char_to_idx.keys())) 
    # import pdb
    # pdb.set_trace()
    print("Chars Stats Done")
    # 设备
    device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")

    # 模型参数
    d_model = args.HIDDEN_SIZE
    nhead = 8
    num_layers = args.LayerNum
    dim_feedforward = args.HIDDEN_SIZE
    max_seq_length = 1024
    dropout = 0.1
    src_vocab_size = dataset.src_vocab_size
    trg_vocab_size = dataset.trg_vocab_size

    model = TransformerSeq2Seq(
        src_vocab_size,
        trg_vocab_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_seq_length,
        dropout
    ).to(device)
    print("Size of Model:",calculate_model_size(model),"MB")
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.char_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs
    clip = 1.0
    
    # 创建 tqdm 实例
    # progress_bar = tqdm(, desc="Training", unit="epoch")
    X_train = tuples
    for epoch in range(num_epochs):
        print("Epoch:",epoch,flush=True)
        # 随机选择索引
        batch_size = args.batch_size
        t0 = time.time()
        indices = random.sample(range(len(X_train)), batch_size*2)
        # 根据索引采样数据
        # 使用 NumPy 的索引操作完成采样
        X_train_sample = [X_train[i] for i in indices]
        Y_train_sample = generateListOfWildCards(X_train_sample , 0.2, insert_tail_prob=0.1, wildcard_prob={'%': args.pct , '_': 1-args.pct})
        
        # import pdb
        # pdb.set_trace()
        datasetTrain = CharacterLevelDataset( Y_train_sample,X_train_sample,)
        batch_size = args.batch_size
        dataloader = DataLoader(datasetTrain, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        start_time = time.time()
        print("SampleTime",start_time-t0)
        train_loss =  train(model, dataloader, optimizer, criterion, clip)
        end_time = time.time()
        print("Train Loss:", train_loss )
        elapsed_time = end_time - start_time
        # 动态更新 tqdm 的描述
        # progress_bar.set_description(f"Epoch {epoch+1}: Loss={train_loss:.3f}, Time={elapsed_time:.2f}s")
        if epoch %(args.IterSave) ==(args.IterSave-1):
            print("saving model")
            torch.save(model,modelSaveRoot+f'/Ep_{epoch}_Seq2Seq')
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
                beamResults =  testBatched(input_texts,model,dataset,K=int(args.inferSampleNum),temperature=0.9)
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
                    print("Sample Result:", "wildcards:",queries[pred_idx], "Value:",results, tRecall, len(results) ,  "Result", candidates[:2],len(candidates))
            print(Recall,"Ratio:",Recall/NumAll)
    print("Seq2Seq Model trained")
    return
    DEVICE = device
    NNCls = TransformerClassifier(model, num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()  # 分类任务常用的交叉熵损失
    optimizer = optim.Adam(NNCls.parameters() )  # 使用
    
    dataloader = prepare_dataCLS(smallQueries , hugeQueries , args.batch_size)
    test_dataloader = dataloader
    for epoch in range(args.epochs):
        NNCls.train()
        total_loss = 0
        for batch_idx, (input_batch, labels) in enumerate(tqdm(dataloader)):
            input_batch, labels = input_batch.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = NNCls(input_batch)
            # import pdb
            # pdb.set_trace()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {total_loss / len(dataloader):.4f}")
        NNCls.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_batch, labels in test_dataloader:
                input_batch, labels = input_batch.to(DEVICE), labels.to(DEVICE)
                outputs = NNCls(input_batch)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{args.epochs}], Test Accuracy: {accuracy:.2f}%")
        if epoch %(args.IterSave) ==(args.IterSave-1):
            print("saving model")
            torch.save(model,modelSaveRoot+f'/Ep_{epoch}_CLS')
           

def load_data(data_path,sampleNum = -1):
    #!! PS ： 师弟你得注意：这个数据的格式必须得是和lineitem的单列格式一样的才行
    if sampleNum==-1:
        print("Using All data as sample")
        t0 = time.time()
        tuples = []
        with open(data_path, 'r') as f:
            tuples = f.readlines()

        retL = [s.replace("\n", '')[:40] if len(s.replace("\n", '')) > 40 else s.replace("\n", '') for s in tqdm(tuples, desc="Processing lines")]
        # 使用 tqdm 显示处理进度
        # retL = [s.replace("\n", '') for s in tqdm(tuples, desc="Processing lines")]
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
    """
    生成一个包含通配符的模式，支持随机跳过字符
    改进点：
    1. 使用索引循环代替字符循环，支持跳过字符
    2. 插入%后可能跳过后续字符
    3. 修复原字符被错误丢弃的问题
    
    参数:
        s: 原始字符串
        insert_char_prob: 在字符后插入通配符的概率
        wildcard_prob: 通配符概率分布
        insert_tail_prob: 尾部插入通配符的概率
        percent_skip_prob: 插入%后跳过后续字符的概率
    """
    pattern = []
    i = 0
    skipEdNum = 0
    while i < len(s):
        # 先添加当前字符（保证字符一定会被包含）
        pattern.append(s[i])
        
        # 判断是否插入通配符
        if random.random() < insert_char_prob:
            wildcard = select_wildcard(wildcard_prob)
            pattern.append(wildcard)
            
            # 如果插入的是%，可能跳过后续字符
            if wildcard == '%' and random.random() < percent_skip_prob:
                # 计算剩余可跳过的字符数（当前字符之后）
                remaining_chars = len(s) - i - 1
                if remaining_chars > 0:
                    # 随机跳过0到remaining_chars个字符
                    skip = random.randint(0, remaining_chars)
                    i += skip  # 直接跳过后续字符
            else:
                i+=1
        
        i += 1
    if skipEdNum ==0:
        random_index = random.randint(0, len(pattern) - 1)
        pattern[random_index]="_"
    return '%'+''.join(pattern)+'%'


def generateListOfWildCards(strList,insert_char_prob=0.5, wildcard_prob={'%': 0.5, '_': 0.5} , insert_tail_prob=0.5):
    """
    输入一个字符串列表，生成等长的通配符，第i个通配符和第i个字符串match
    """ 
    wildStrings = []
    for si in tqdm(strList, desc="Processing", unit="item"):
        wildStrings.append(generate_wildcard_pattern( si,insert_char_prob, wildcard_prob, insert_tail_prob))
    return wildStrings
def profile_list_of_lists(nested_list):
    # 创建一个字典来统计每个长度的出现次数
    length_distribution = {}

    # 遍历嵌套列表中的每个子列表
    for sublist in nested_list:
        # 获取子列表的长度
        sublist_length = len(sublist[1])
        # 如果长度已经在字典中，计数加1；否则初始化为1
        if sublist_length in length_distribution:
            length_distribution[sublist_length] += 1
        else:
            length_distribution[sublist_length] = 1

    return length_distribution

# 设置随机种子的函数
def setup_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = False     # 关闭加速优化
if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Train a seq2seq model with attention.")
    parser.add_argument('--data_path', type=str, default="./data/lineitem.csv", help='Path to data file.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train the model.')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1, help='Teacher forcing ratio for training.')
    parser.add_argument('--max_length', type=int, default=40, help='Maximum length for output sequences.')
    parser.add_argument('--HIDDEN_SIZE', type=int, default=512)
    parser.add_argument('--LayerNum', type=int, default=1)
    parser.add_argument('--GPU', type=int, default=5)
    parser.add_argument('--inferSampleNum', type=int, default= 4, help='Number of Inference Samples')
    parser.add_argument('--inferParrellism', type=int, default=64, help='Inference Paralism')
    parser.add_argument('--lr', type=float, default=0.0004, help='Number of samples for inference.')
    parser.add_argument('--seed', type=float, default=42, help='Number of samples for inference.')
    parser.add_argument('--saveName', type=str, default="tmpModel", help='model save folder')
    parser.add_argument('--IterSave', type=int, default= 500, help='Number of Iter Saving Model and Eval')
    parser.add_argument('--pct', type=float, default= 0.2, help='''Ratio of \% in workloads''')
    parser.add_argument('--inPct', type=float, default= 0.1, help='Path Of Evaling Model')
    parser.add_argument('--numTem', type=int, default= -1, help='Number of temtemplate')
    parser.add_argument('--numRepeat', type=int, default= 1, help='Number of temtemplate seedOrigin')
    args = parser.parse_args()
    setup_seed(args.seed)
    
    smallQueries,hugeQueries = [],[]
    print_args(args)
    seq2seqTransformer(args,smallQueries,hugeQueries ) 
