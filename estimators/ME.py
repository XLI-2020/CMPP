import numpy as np
from utils.data_process import train_test_valid_split,norm_Adj
import torch
import time
from torch import optim
from datetime import datetime
from utils.model_component import integrated_loss,ParalleStGcn, eval_kl, eval_dist
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


parser = argparse.ArgumentParser(description='ME')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--time_interval', type=str, default='five_mins')
args = parser.parse_args()
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

st = datetime.now()
print('start at:', st)
t = args.time_interval

features = np.load('./input/BLD-1/{t1}/features_{t2}.npy'.format(t1=t, t2=t.split('_')[0]))
# np.save('./input/MallB/{t1}/features_{t2}.npy'.format(t1=t, t2=t.split('_')[0]), features[:1692,:,:,:])
labels = np.load('./input/BLD-1/{t1}/labels_{t2}.npy'.format(t1=t, t2=t.split('_')[0]))
# np.save('./input/MallB/{t1}/labels_{t2}.npy'.format(t1=t, t2=t.split('_')[0]), labels[:1692,:,:,:])

print(features.shape)
print(labels.shape)
features, labels, idx_train, idx_validation, idx_test = train_test_valid_split(features, labels, train_rate=0.7, validation_rate=0.1)
adj = np.load('./input/BLD-1/Adj.npy')
print('Adj shape',adj.shape)
norm_adj = norm_Adj(adj)
features = torch.FloatTensor(features)
features = torch.transpose(features,1,2)
labels = torch.FloatTensor(labels).to(device=args.device)
labels = torch.transpose(labels,1,2).to(device=args.device)
norm_adj = torch.FloatTensor(norm_adj).to(device=args.device)
in_channels = 2
out_channels = 16
hidden_size = 16
output_size = 1
LR = 0.01

model = ParalleStGcn(norm_adj,in_channels,out_channels,hidden_size,output_size).to(device=args.device)
optimizer = optim.Adam(model.parameters(),lr=LR)
batch_size = args.batch_size
batches = int(len(idx_train)/batch_size)


def label_transform(idx_train):
    label1, label2 = labels[idx_train][:, :, :, 0], labels[idx_train][:, :, :, 1]
    label1 = torch.reshape(label1, (-1, output_size))
    label2 = torch.reshape(label2, (-1, output_size))
    return label1, label2

def train(epoch):
    model.train()
    for batch in range(batches):
        optimizer.zero_grad()
        batch_idx = idx_train[batch_size*batch:batch_size*(batch+1)]
        out1, out2 = model(features[batch_idx].to(device=args.device))
        label1, label2 = label_transform(batch_idx)
        label1 = label1.to(device=args.device)
        label2 = label2.to(device=args.device)
        loss = integrated_loss(out1,out2,label1,label2)
        loss.backward()
        optimizer.step()
    model.train(False)
    with torch.no_grad():
        out_val1, out_val2 = model(features[idx_validation].to(device=args.device))
        label_val1, label_val2 = label_transform(idx_validation)
        label_val1 = label_val1.to(device=args.device)
        label_val2 = label_val2.to(device=args.device)
        loss_val = integrated_loss(out_val1, out_val2, label_val1, label_val2)
        eu_dist = eval_dist(out_val1.cpu().numpy(), out_val2.cpu().numpy(), label_val1.cpu().numpy(),
                            label_val2.cpu().numpy())
        kl = eval_kl(out_val1.cpu().numpy(), out_val2.cpu().numpy(), label_val1.cpu().numpy(),
                     label_val2.cpu().numpy())

    print('Epoch:{:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'euclidean dist: {:.4f}'.format(eu_dist),
          'kl: {:.4f}'.format(kl))
    return eu_dist, kl

def test():
    model.eval()
    out_val1, out_val2 = model(features[idx_test].to(device=args.device))
    label_val1, label_val2 = label_transform(idx_test)
    label_val1 = label_val1.to(device=args.device)
    label_val2 = label_val2.to(device=args.device)
    loss_test = integrated_loss(out_val1, out_val2, label_val1, label_val2)
    eu_dist = eval_dist(out_val1.cpu().numpy(), out_val2.cpu().numpy(), label_val1.cpu().numpy(),
                        label_val2.cpu().numpy())
    kl = eval_kl(out_val1.cpu().numpy(), out_val2.cpu().numpy(), label_val1.cpu().numpy(),
                 label_val2.cpu().numpy())
    print('Test:\n',
          'loss_test: {:.4f}'.format(loss_test.item()),
          'euclidean dist: {:.4f}'.format(eu_dist),
          'kl: {:.4f}'.format(kl))




train_epochs = args.epochs
t_total = time.time()
epoch_results = []
time_eva = []
for epoch in range(train_epochs):
    e_dist, kl = train(epoch)

test()

et = datetime.now()
print('end at:', et)

dt = (et - st).total_seconds() / 3600  # hours
print('elapsed time:', str(dt))


