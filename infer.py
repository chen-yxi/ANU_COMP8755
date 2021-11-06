import torch
import torch.nn as nn
import numpy as np
import os
from train import split_data_by_windows,get_data,gen_nopeek_mask
from einops import rearrange
from matplotlib import pyplot
from model import Translation
from model import MyDataset
from config import params

# model = torch.load(params['model_path'])
model = torch.load('models\\transformer.pth')

len_seq = params['input_window']
batch_size = params['batch_size']
criterion = nn.MSELoss()
HR,HR_t,BVP,BVP_t,EDA,EDA_t = split_data_by_windows(len_seq)
train_dict = {'HR':HR   ,'BVP':BVP   ,'EDA':EDA  }
test_dict  = {'HR':HR_t ,'BVP':BVP_t ,'EDA':EDA_t}

# sampels = int(len(HR)*0.8)
# train_data_src = HR[:sampels]
# train_data_tgt = EDA[:sampels]

# test_data_src = HR[sampels:]
# test_data_tgt = EDA[sampels:]

train_data_src = train_dict[params['src']]
train_data_tgt = train_dict[params['tgt']]

test_data_src = test_dict[params['src']]
test_data_tgt = test_dict[params['tgt']]

print(len(test_data_src),len(test_data_tgt))

predicted = np.array([])
total_loss = 0


for i in range(0,len(test_data_src),batch_size):
    src = torch.tensor(test_data_src[i:i+batch_size]).long().to('cuda')
    tgt = torch.tensor(test_data_tgt[i:i+batch_size]).to('cuda')
    tgt_mask = gen_nopeek_mask(tgt.shape[1]).to('cuda')
    output = model.forward(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=tgt_mask)
    loss = criterion(rearrange(output, 'b t v -> (b t v)').to(torch.float32), rearrange(tgt, 'b o -> (b o)').to(torch.float32))
    total_loss += loss.item()
    output = rearrange(output, 'b t v -> (b t v)').to(torch.float32)
    output = output.data.cpu().numpy()
    predicted = np.hstack((predicted,output))
    
    # print(np.shape(output),type(output))
mean_loss = total_loss/(i//batch_size+1)
print("loss:",mean_loss)
target = np.array(test_data_tgt).reshape(1,-1).tolist()[0]
diff = list(map(lambda x: x[0]-x[1], zip(target, predicted)))
seq = params['seq']

if not os.path.exists(params['result_path']):
    os.mkdir(params['result_path'])


for step in range(0,len(test_data_tgt)*len_seq*params['Freq'],seq):
    pyplot.plot(predicted[step:step+seq],color="red",label='Output')       
    pyplot.plot(target[step:step+seq],color="blue",label='Target')
    pyplot.plot(diff[step:step+seq],color="green",label='Diff')
    pyplot.legend()
    # pyplot.grid(True, which='both')
    # pyplot.axhline(y=0, color='k')
    pyplot.savefig('{0}2{1}results{2}/{1}{3}.png'.format(params['src'],params['tgt'],seq,step))
    pyplot.close()
