import numpy as np
from utils.data_process import train_test_valid_split,norm_Adj
from utils.model_component import ParalleStGcn
import torch
import torch.nn as nn
import math
import time
from torch import optim
from datetime import datetime
from utils.model_component import SaGcnGru,Gru, integrated_loss,ParalleStGcn, eval_kl, eval_dist, kl_loss, waserstein_loss
import argparse
parser = argparse.ArgumentParser(description='ME')
args = parser.parse_args()
np.random.seed(2021)
st = datetime.now()

if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
print('start at:', st)
t = 'one_min'
features = np.load('./input/MallB/{t1}/features_{t2}.npy'.format(t1=t, t2=t.split('_')[0]))
labels = np.load('./input/MallB/{t1}/labels_{t2}.npy'.format(t1=t, t2=t.split('_')[0]))
print(features.shape)
print(labels.shape)
features, labels, idx_train, idx_validation, idx_test = train_test_valid_split(features, labels, train_rate=0.7, validation_rate=0.1)
adj = np.load('./input/MallB/Adj.npy')
print('Adj shape',adj.shape)
norm_adj = norm_Adj(adj)
features = torch.FloatTensor(features)
features = torch.transpose(features,1,2).to(device=args.device)
labels = torch.FloatTensor(labels)
labels = torch.transpose(labels,1,2).to(device=args.device)
norm_adj = torch.FloatTensor(norm_adj).to(device=args.device)
in_channels = 2
out_channels = 16
hidden_size = 16
output_size = 1
LR = 0.01

model = ParalleStGcn(norm_adj,in_channels,out_channels,hidden_size,output_size).to(device=args.device)
optimizer = optim.Adam(model.parameters(),lr=LR)
batch_size = 64
batches = int(len(idx_train)/batch_size)


def label_transform(idx_train):
    label1, label2 = labels[idx_train][:, :, :, 0], labels[idx_train][:, :, :, 1]
    label1 = torch.reshape(label1, (-1, output_size))
    label2 = torch.reshape(label2, (-1, output_size))
    return label1, label2

def train(epoch):
    t = time.time()
    model.train()
    for batch in range(batches):
        optimizer.zero_grad()
        batch_idx = idx_train[batch_size*batch:batch_size*(batch+1)]
        out1, out2 = model(features[batch_idx])
        label1, label2 = label_transform(batch_idx)
        loss = waserstein_loss(out1,out2,label1,label2)
        # loss = kl_loss(out1,out2,label1,label2)
        loss.backward()
        optimizer.step()
    model.eval()
    vl = len(idx_validation)
    vst = time.time()
    out_val1, out_val2 = model(features[idx_validation])
    vet = time.time()
    label_val1, label_val2 = label_transform(idx_validation)
    loss_val = waserstein_loss(out_val1, out_val2, label_val1, label_val2)
    # loss_val = kl_loss(out_val1, out_val2, label_val1, label_val2)
    out_val = torch.cat([out_val1,out_val2],dim=0)
    label_val = torch.cat([label_val1,label_val2],dim=0)
    eu_dist = eval_dist(out_val1.detach().numpy(), np.square(out_val2.detach().numpy()), label_val1.detach().numpy(), label_val2.detach().numpy())
    kl = eval_kl(out_val1.detach().numpy(), np.square(out_val2.detach().numpy()), label_val1.detach().numpy(), label_val2.detach().numpy())
    print('Epoch:{:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'euclidean dist: {:.4f}'.format(eu_dist),
          'kl: {:.4f}'.format(kl))
    return eu_dist, kl

def evaluate():
    model.eval()
    out_val1, out_val2 = model(features[idx_test])
    label_val1, label_val2 = label_transform(idx_test)
    loss_test = waserstein_loss(out_val1, out_val2, label_val1, label_val2)
    eu_dist = eval_dist(out_val1.detach().numpy(), np.square(out_val2.detach().numpy()), label_val1.detach().numpy(),
                        label_val2.detach().numpy())
    kl = eval_kl(out_val1.detach().numpy(), np.square(out_val2.detach().numpy()), label_val1.detach().numpy(),
                 label_val2.detach().numpy())

    print('Test:\n',
          'loss_test: {:.4f}'.format(loss_test.item()),
          'euclidean dist: {:.4f}'.format(eu_dist),
          'kl: {:.4f}'.format(kl))


train_epochs = 50
fine_train_epochs = 10
t_total = time.time()
epoch_results = []
time_eva = []
for epoch in range(train_epochs):
    e_dist, kl = train(epoch)

evaluate()

et = datetime.now()
print('end at:', et)

dt = (et - st).total_seconds() / 3600  # hours
print('elapsed time:', str(dt))


