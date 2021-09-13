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

import pretrain

model_path = 'Pretrained_model.pth'
model_pretrain = models_train.MLP(h_list=[2048, 2048, 1024], out_featrues=229, mode='pre')
model_pretrain.to(device)
model_trans = nn.DataParallel(model_pretrain, device_ids = [2, 0, 1])
model_trans.load_state_dict(torch.load(model_path))
#Retrain the classifier layer
#model_trans.module.linears[-1] = nn.Linear(1024, 229).to(device)
hot_params = list(map(id, model_trans.module.linears[-1].parameters()))
froze_params = filter(lambda p: id(p) not in hot_params, model_trans.module.parameters())

for p in froze_params:
    p.requires_grad = False
lr = 1e-3
optimizer = torch.optim.Adam(model_trans.parameters(), lr=lr, weight_decay=0.002, betas=(0.9, 0.999))
train_data = models_train.netdata('train_cc_ecfp.csv', 
                  'train_cc_activity.csv', 
                   'train_cc_weight.csv', 'index_Lthan10.csv')
train_data_loader = DataLoader(train_data, shuffle=True, batch_size=32)
test_data = models_train.netdata('test_cc_ecfp.csv', 
                  'test_cc_activity.csv', 
                   'test_cc_weight.csv',  'index_Lthan10.csv')
test_data_loader = DataLoader(test_data, batch_size=256)

au_roc = []
for epoch in range(200):
     if epoch % 20 == 0:
        with torch.no_grad():
            model_trans.eval()
            for cat_c, test_data_batch in enumerate(test_data_loader):
                batch = test_data_batch[0].to(device).float()
                test_output_batch = model_trans(batch)
                
                test_label_batch = test_data_batch[1]
                if cat_c == 0:
                    test_output = test_output_batch.cpu()
                    test_label = test_label_batch.cpu()
                else:
                    test_output = torch.cat((test_output, test_output_batch.cpu()), dim=0)
                    test_label = torch.cat((test_label, test_label_batch.cpu()), dim=0)
            
            fpr, tpr, _ = metrics.roc_curve(test_label.view(-1).detach().numpy(), test_output.view(-1).detach().numpy(),  pos_label=1 )
            au_roc.append(metrics.auc(fpr, tpr))
            
            print(' \r  training at epoch %d now have auroc %f '%( epoch, au_roc[-1]))


     for batch in train_data_loader:
        model_trans.train()
        input_batch, label_batch = batch[0].to(device).float(), batch[1].to(device).float()
        optimizer.zero_grad()
        outputs = model_trans(input_batch)
        

        loss = F.binary_cross_entropy(outputs, label_batch, weight=batch[2].to(device).float())
        loss.backward()
        optimizer.step()
     print(f'\rtask epoch {epoch} running', end=' ') 

