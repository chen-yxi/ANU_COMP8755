import torch
import torch.nn as nn
import numpy as np
import time
import math
import random
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm
from matplotlib import pyplot
from einops import rearrange
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Optim import ScheduledOptim

from model import Translation
from model import MyDataset
from config import params

torch.manual_seed(0)
np.random.seed(0)

# This concept is also called teacher forceing. 
# The flag decides if the loss will be calculted over all 
# or just the predicted values.

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)
#
#print(out)

input_window = params['input_window']
output_window = params['output_window']
batch_size = params['batch_size'] # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_norm(s,reverse=False):
        s = (s - np.mean(s))/np.std(s)
        if reverse:
            s = -s
        return s

def split_data(data,len_seq):
    splited = []
    for i in range(len(data)//len_seq):
        splited.append(list(data[i*len_seq:(i+1)*len_seq]))
    if not len(data)%len_seq:
        splited.append(list(data[(i+1)*len_seq:]))
    return splited


def split_data_by_windows(len_seq):
    H = []
    H_t = []
    B = []
    B_t = []
    E = []
    E_t = []
    for i in range(1,50):
        ind = str(i).zfill(2)
        BVP = "data/p"+ind+"/BVP.csv"
        HR = "data/p"+ind+"/HR.csv"
        EDA = "data/p"+ind+"/EDA.csv"

        bvp = pd.read_csv(BVP)
        hr = pd.read_csv(HR)
        eda = pd.read_csv(EDA)

        h = hr.iloc[:,0].values[1:]
        b = bvp.iloc[:,0].values[641:]
        e = eda.iloc[:,0].values[41:] 

        h = data_norm(h)
        b = data_norm(b)
        e = data_norm(e)

        min_len = min(len(h),len(b)//64,len(e)//4)
        min_train_len = int(0.8*min_len)

        # H += split_data(h[:min_len],len_seq)
        # B += split_data(b[:64*min_len],64*len_seq)
        # E += split_data(e[:4*min_len],4*len_seq)
        H += list(h[:min_train_len]) 
        B += list(b[:64*min_train_len])
        E += list(e[:4*min_train_len])
        H_t += list(h[min_train_len:min_len])
        B_t += list(b[64*min_train_len:64*min_len])
        E_t += list(e[4*min_train_len:4*min_len])

    H = split_data(H,len_seq)
    H_t = split_data(H_t,len_seq)
    B = split_data(B,64*len_seq)
    B_t = split_data(B_t,64*len_seq)
    E = split_data(E,4*len_seq)
    E_t = split_data(E_t,4*len_seq)

    return H,H_t,B,B_t,E,E_t
    # print(type(H))
    # return H,B,E


def get_data(len_seq,batch_size):
   
    HR,HR_t,BVP,BVP_t,EDA,EDA_t = split_data_by_windows(len_seq) #shape: N, S
    train_dict = {'HR':HR   ,'BVP':BVP   ,'EDA':EDA  }
    test_dict  = {'HR':HR_t ,'BVP':BVP_t ,'EDA':EDA_t}
    # sampels = int(len(HR)*0.8)
    # train_data_src = HR[:sampels]
    # train_data_tgt = EDA[:sampels]

    # test_data_src = HR[sampels:]
    # test_data_tgt = EDA[sampels:]
    '''
    !!!Select src signal and tgt by modify here!!!
    '''
    train_data_src = train_dict[params['src']]
    train_data_tgt = train_dict[params['tgt']]

    test_data_src = test_dict[params['src']]
    test_data_tgt = test_dict[params['tgt']]


    train_dataset = MyDataset(train_data_src,train_data_tgt,batch_size,vali=False)
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=4,pin_memory=True)
    test_dataset = MyDataset(test_data_src,test_data_tgt,batch_size,vali=True)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4,pin_memory=True)

    return train_loader,test_loader

def train(train_loader, valid_loader, model, optim, criterion, num_epochs):
    pbarlen = len(train_loader)
    model.train() # Turn on the train mode
    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0

    for epoch in range(num_epochs):
        pbar = tqdm(total=pbarlen, leave=False)
        total_loss = 0

        # Shuffle batches every epoch
        train_loader.dataset.shuffle_batches()
        for step, (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in enumerate(iter(train_loader)):
            total_step += 1

            # Send the batches and key_padding_masks to gpu
            src, src_key_padding_mask = src[0].to(device), src_key_padding_mask[0].to(device)
            tgt, tgt_key_padding_mask = tgt[0].to(device), tgt_key_padding_mask[0].to(device)
            memory_key_padding_mask = src_key_padding_mask.clone()

            # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)
            # tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
            # tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(device)
            tgt_mask = gen_nopeek_mask(tgt.shape[1]).to(device)
            # Forward
            optim.zero_grad()
            

            outputs = model(src, tgt, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask,tgt_mask)
            loss = criterion(rearrange(outputs, 'b t v -> (b t v)').to(torch.float32), rearrange(tgt, 'b o -> (b o)').to(torch.float32))


            # Backpropagate and update optim
            loss.backward()
            optim.step_and_update_lr()

            total_loss += loss.item()
            train_losses.append((step, loss.item()))
            pbar.update(1)
            log_interval = int(len(train_loader) / batch_size / 5)
            
            if step % pbarlen == pbarlen - 1:
                pbar.close()
                print(f'Epoch [{epoch + 1} / {num_epochs}] \t'
                      f'Train Loss: {total_loss / pbarlen}')
                total_loss = 0

                pbar = tqdm(total=pbarlen, leave=False)

        '''
        !!!Modify the model name!!!
        '''
        # Validate every epoch
        pbar.close()
        val_loss = validate(valid_loader, model, criterion)
        val_losses.append((total_step, val_loss))
        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model,params['model_path'])
        print(f'Val Loss: {val_loss}')
    return train_losses, val_losses

def validate(valid_loader, model, criterion):
    pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.eval()

    total_loss = 0
    for step, (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in enumerate(iter(valid_loader)):
        # print("shape:",src.shape,src_key_padding_mask.shape,tgt.shape,tgt_key_padding_mask.shape)
        # print("shape:",src[0].shape,tgt[0].shape)
        with torch.no_grad():
            src, src_key_padding_mask = src[0].to(device), src_key_padding_mask[0].to(device)
            tgt, tgt_key_padding_mask = tgt[0].to(device), tgt_key_padding_mask[0].to(device)
            memory_key_padding_mask = src_key_padding_mask.clone()
            # tgt_inp = tgt[:, :-1]
            # tgt_out = tgt[:, 1:].contiguous()
            # tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(device)
            tgt_mask = gen_nopeek_mask(tgt.shape[1]).to(device)

            outputs = model(src, tgt, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask,tgt_mask)
            loss = criterion(rearrange(outputs, 'b t v -> (b t v)').to(torch.float32), rearrange(tgt, 'b o -> (b o)').to(torch.float32))
            total_loss += loss.item()
            pbar.update(1)

    pbar.close()
    model.train()
    return total_loss / len(valid_loader)

    
def gen_nopeek_mask(length):
    """
     Returns the nopeek mask
             Parameters:
                     length (int): Number of tokens in each sentence in the target batch
             Returns:
                     mask (arr): tgt_mask, looks like [[0., -inf, -inf],
                                                      [0., 0., -inf],
                                                      [0., 0., 0.]]
     """
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask

if __name__ == '__main__':
    train_loader, valid_loader = get_data(input_window,batch_size)
    model = Translation().to(device)

    criterion = nn.MSELoss()
    lr = 0.001 
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optim = ScheduledOptim(
        Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),250,4000)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

    best_val_loss = float("inf")
    epochs = 20 # The number of epochs
    best_model = None

    train_losses, val_losses = train(train_loader, valid_loader, model, optim, criterion, epochs)

