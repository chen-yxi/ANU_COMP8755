import torch
import torch.nn as nn
import numpy as np
import random
import math
from torch.utils.data import Dataset
from einops import rearrange


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
       

class Translation(nn.Module):
    def __init__(self,feature_size=250,num_encoder_layers=3,num_decoder_layers=3,dropout=0.1):
        super(Translation, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.tgt_mask = None
        self.pos_encoder = PositionalEncoding(d_model=feature_size, dropout=dropout)
        self.encoder_layer = nn.Transformer(d_model=feature_size, nhead=10,num_encoder_layers= num_encoder_layers,num_decoder_layers=num_decoder_layers ,dropout=dropout)      
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src,tgt,src_key_padding_mask,tgt_key_padding_mask, memory_key_padding_mask,tgt_mask):

        src = rearrange(src, '(n k) s -> s n k',k=1).to(torch.float32)
        tgt = rearrange(tgt, '(n k) t -> t n k',k=1).to(torch.float32)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.encoder_layer(src, tgt, tgt_mask=self.tgt_mask, src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = rearrange(output, 't n e -> n t e')
        output = self.decoder(output)
        return output

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

class MyDataset(Dataset):
    def __init__(self, src, tgt, batch_size, vali):
        self.src = src 
        self.tgt = tgt
        self.vali = vali
        self.batch_size = batch_size
        self.batches = batch_init(len(self.src),self.batch_size)

    def __getitem__(self, idx):
        src, src_mask = getitem(idx, self.src, self.batches)
        tgt, tgt_mask = getitem(idx, self.tgt, self.batches)
        return src, src_mask, tgt, tgt_mask

    def __len__(self):
        return len(self.batches)

    def shuffle_batches(self):
        self.batches = batch_init(len(self.src),self.batch_size)

def getitem(idx,data,batches):
    index = batches[idx]
    batch = [data[i] for i in index]

    # Get the maximum sentence length of this batch
    seq_length = 0
    for sentence in batch:
        if len(sentence) > seq_length:
            seq_length = len(sentence)

    masks = []
    for i, sentence in enumerate(batch):
        # Generate the masks for each sentence, False if there's a token, True if there's padding
        masks.append([False for _ in range(len(sentence))] + [True for _ in range(seq_length - len(sentence))])
        # Add 0 padding
        batch[i] = sentence + [0 for _ in range(seq_length - len(sentence))]
    # if src:
    #     print(index)
    return np.array(batch), np.array(masks)

def batch_init(l,batch_size):
    batches = []
    tmp = [i for i in range(l)]
    random.shuffle(tmp)
    #Spilt index by batch size
    for i in range(l//batch_size):
        batches.append(tmp[batch_size*i:batch_size*(i+1)])
    if l%batch_size:
        batches.append(tmp[batch_size*(i+1):])
    
    return batches