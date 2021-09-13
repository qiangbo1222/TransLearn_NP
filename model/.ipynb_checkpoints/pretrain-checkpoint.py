import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class netdata(Dataset):
    def __init__(self, fingerprint_path, index_path, label_path=None, weight_path=None, label_num=229, mode='source'):
        self.mode = mode
        print(mode)
        list_use_target_head0 = list(pd.read_csv(index_path, index_col=0).iloc[:, 0].values)
        if self.mode == 'source' or self.mode == 'test':
            self.ecfp4 = np.array(pd.read_csv(fingerprint_path, index_col=0))
            self.label_all = np.array(pd.read_csv(label_path, index_col=0, usecols=list_use_target_head0))
            self.weight = np.array(pd.read_csv(weight_path, index_col=0))
            if self.label_all.shape[0] != self.ecfp4.shape[0]:
                print(self.label_all.shape[0])
                print(self.ecfp4.shape[0])
                raise  ValueError('incompatible shape of label and fingerprint')
        
        else:
            self.ecfp4 = np.array(pd.read_csv(fingerprint_path, index_col=0))
    
    def __len__(self):
        return self.ecfp4.shape[0]
    
    def __getitem__(self, idx):
        if self.mode == 'source' or self.mode == 'test':
            input_tensor = torch.tensor(self.ecfp4[idx, ...])
            label_tensor = torch.tensor(self.label_all[idx, ...])
            weight_tensor = torch.tensor(self.weight[idx, ...])
            return input_tensor, label_tensor, weight_tensor

        else:
            input_tensor = torch.tensor(self.ecfp4[idx, ...])
            return input_tensor


class MLP(nn.Module):
    def __init__(self, h_list, out_featrues=229, mode='pre'):
        super(MLP, self).__init__()
        h_list1, h_list2 = h_list.copy(), h_list.copy()
        h_list2.append(out_featrues)
        h_list1.insert(0, 2048)
        self.linears = nn.ModuleList([nn.Linear(x, y) for x, y in zip(h_list1 , h_list2)])
        self.BNs = nn.ModuleList([nn.BatchNorm1d(m) for m in h_list])
        self.h_list = h_list
        self.mode = mode
        
    
    def forward(self, x):
        x = x.float()
       
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            x = self.BNs[i](x)
            x = F.leaky_relu(x)
        
        if self.mode == 'trans_DDC':
            pass
        else :
            x = torch.sigmoid(self.linears[-1](x))
        return x
