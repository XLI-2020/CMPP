import numpy as np
import sys
sys.path.append('/Users/xiaol/PycharmProjects/CMPP')
from utils.data_process import train_test_valid_split,norm_Adj
import torch
import torch.nn as nn
import math
import time
from datetime import datetime
from torch import optim
from utils.model_component import integrated_loss,eval_kl, eval_dist, GruMulti, waserstein_loss

np.random.seed(2021)

st = datetime.now()
print('start at:', st)
for t in ['one_min', 'five_mins', 'ten_mins']:
    print('!!!!frequency:', t)
    features = np.load('../input/MallA/{t1}/features_{t2}.npy'.format(t1=t, t2=t.split('_')[0]))
    labels = np.load('../input/MallA/{t1}/labels_{t2}.npy'.format(t1=t, t2=t.split('_')[0]))
    print(features.shape)
    print(labels.shape)


    features, labels, idx_train, idx_validation, idx_test = train_test_valid_split(features, labels, train_rate=0.7, validation_rate=0.1)


    adj = np.load('../input/MallA/Adj.npy')
    print(adj.shape)
    norm_adj = norm_Adj(adj)

    features = torch.FloatTensor(features)
    features = torch.transpose(features,1,2)
    labels = torch.FloatTensor(labels)
    labels = torch.transpose(labels,1,2)

    norm_adj = torch.FloatTensor(norm_adj)
    in_channels = 2
    out_channels = 16
    hidden_size = 16
    output_size = 1
    LR = 0.01

    model = GruMulti(in_channels,hidden_size,output_size)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    batch_size = 64
    batches = int(len(idx_train)/batch_size)
    time_interval = 'ten_secs'#1min, 5mins


    def label_transform(idx_train):
        label1, label2 = labels[idx_train][:, :, :, 0], labels[idx_train][:, :, :, 1]
        label1 = torch.reshape(label1, (-1, output_size))
        label2 = torch.reshape(label2, (-1, output_size))
        return label1, label2

    # Model_save_path = './se_model.pkl'
    def train(epoch):
        t = time.time()
        model.train()
        for batch in range(batches):
            optimizer.zero_grad()
            batch_idx = idx_train[batch_size*batch:batch_size*(batch+1)]
            out1, out2 = model(features[batch_idx])
            label1, label2 = label_transform(batch_idx)
            loss = waserstein_loss(out1,out2,label1,label2)
            loss.backward()
            optimizer.step()
        model.eval()
        vl = len(idx_validation)
        vst = time.time()
        out_val1, out_val2 = model(features[idx_validation])
        vet = time.time()
        label_val1, label_val2 = label_transform(idx_validation)
        loss_val = waserstein_loss(out_val1, out_val2, label_val1, label_val2)
        out_val = torch.cat([out_val1,out_val2],dim=0)
        label_val = torch.cat([label_val1,label_val2],dim=0)

        eu_dist = eval_dist(out_val1.detach().numpy(), np.square(out_val2.detach().numpy()), label_val1.detach().numpy(),
                            label_val2.detach().numpy())
        kl = eval_kl(out_val1.detach().numpy(), np.square(out_val2.detach().numpy()), label_val1.detach().numpy(),
                     label_val2.detach().numpy())

        print('Epoch:{:04d}'.format(epoch + 1),
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


    train_epochs = 100
    fine_train_epochs = 10
    t_total = time.time()
    epoch_results = []
    time_eva = []

    for epoch in range(train_epochs):
        e_dist, kl = train(epoch)


    evaluate()

    break

et = datetime.now()

print('end at:', et)

dt = (et-st).total_seconds()/3600 #hours
print('elapsed time:', str(dt))



"""

start at: 2024-07-06 09:35:16.211259
!!!!frequency: one_min
(12980, 10, 300, 2)
(12980, 1, 300, 2)
len(idx_validation) 1298
len(idx_test) 2596
(300, 300)
Epoch:0001 loss_train: 5.3343 loss_val: 6.5194 euclidean dist: 0.7982 kl: 3.9172
Epoch:0002 loss_train: 2.8209 loss_val: 3.7801 euclidean dist: 0.7168 kl: 1.8215
Epoch:0003 loss_train: 1.7624 loss_val: 2.4374 euclidean dist: 0.6743 kl: 1.3344
Epoch:0004 loss_train: 1.1777 loss_val: 1.6115 euclidean dist: 0.6570 kl: 1.1315
Epoch:0005 loss_train: 0.9499 loss_val: 1.2328 euclidean dist: 0.6471 kl: 1.0571
Epoch:0006 loss_train: 0.8454 loss_val: 1.0403 euclidean dist: 0.6341 kl: 1.0118
Epoch:0007 loss_train: 0.7957 loss_val: 0.9290 euclidean dist: 0.6235 kl: 0.9840
Epoch:0008 loss_train: 0.7686 loss_val: 0.8614 euclidean dist: 0.6136 kl: 0.9661
Epoch:0009 loss_train: 0.7520 loss_val: 0.8190 euclidean dist: 0.6085 kl: 0.9540
Epoch:0010 loss_train: 0.7420 loss_val: 0.7898 euclidean dist: 0.6052 kl: 0.9479
Epoch:0011 loss_train: 0.7364 loss_val: 0.7662 euclidean dist: 0.6075 kl: 0.9452
Epoch:0012 loss_train: 0.7303 loss_val: 0.7520 euclidean dist: 0.6106 kl: 0.9479
Epoch:0013 loss_train: 0.7254 loss_val: 0.7415 euclidean dist: 0.6117 kl: 0.9498
Epoch:0014 loss_train: 0.7198 loss_val: 0.7332 euclidean dist: 0.6118 kl: 0.9509
Epoch:0015 loss_train: 0.7155 loss_val: 0.7259 euclidean dist: 0.6099 kl: 0.9484
Epoch:0016 loss_train: 0.7130 loss_val: 0.7195 euclidean dist: 0.6067 kl: 0.9429
Epoch:0017 loss_train: 0.7107 loss_val: 0.7149 euclidean dist: 0.6049 kl: 0.9396
Epoch:0018 loss_train: 0.7088 loss_val: 0.7115 euclidean dist: 0.6041 kl: 0.9381
Epoch:0019 loss_train: 0.7071 loss_val: 0.7087 euclidean dist: 0.6036 kl: 0.9372
Epoch:0020 loss_train: 0.7056 loss_val: 0.7065 euclidean dist: 0.6032 kl: 0.9370
Epoch:0021 loss_train: 0.7042 loss_val: 0.7047 euclidean dist: 0.6029 kl: 0.9374
Epoch:0022 loss_train: 0.7028 loss_val: 0.7031 euclidean dist: 0.6026 kl: 0.9380
Epoch:0023 loss_train: 0.7015 loss_val: 0.7019 euclidean dist: 0.6024 kl: 0.9388
Epoch:0024 loss_train: 0.7005 loss_val: 0.7009 euclidean dist: 0.6021 kl: 0.9382
Epoch:0025 loss_train: 0.6996 loss_val: 0.6999 euclidean dist: 0.6018 kl: 0.9375
Epoch:0026 loss_train: 0.6988 loss_val: 0.6991 euclidean dist: 0.6013 kl: 0.9366
Epoch:0027 loss_train: 0.6982 loss_val: 0.6984 euclidean dist: 0.6009 kl: 0.9358
Epoch:0028 loss_train: 0.6976 loss_val: 0.6978 euclidean dist: 0.6004 kl: 0.9352
Epoch:0029 loss_train: 0.6971 loss_val: 0.6973 euclidean dist: 0.6000 kl: 0.9347
Epoch:0030 loss_train: 0.6967 loss_val: 0.6968 euclidean dist: 0.5997 kl: 0.9343
Epoch:0031 loss_train: 0.6963 loss_val: 0.6963 euclidean dist: 0.5994 kl: 0.9340
Epoch:0032 loss_train: 0.6961 loss_val: 0.6959 euclidean dist: 0.5992 kl: 0.9338
Epoch:0033 loss_train: 0.6958 loss_val: 0.6955 euclidean dist: 0.5991 kl: 0.9338
Epoch:0034 loss_train: 0.6956 loss_val: 0.6951 euclidean dist: 0.5989 kl: 0.9340
Epoch:0035 loss_train: 0.6955 loss_val: 0.6947 euclidean dist: 0.5988 kl: 0.9344
Epoch:0036 loss_train: 0.6953 loss_val: 0.6944 euclidean dist: 0.5986 kl: 0.9349
Epoch:0037 loss_train: 0.6951 loss_val: 0.6939 euclidean dist: 0.5986 kl: 0.9358
Epoch:0038 loss_train: 0.6950 loss_val: 0.6934 euclidean dist: 0.5981 kl: 0.9363
Epoch:0039 loss_train: 0.6948 loss_val: 0.6925 euclidean dist: 0.5990 kl: 0.9389
Epoch:0040 loss_train: 0.6949 loss_val: 0.6918 euclidean dist: 0.5969 kl: 0.9384
Epoch:0041 loss_train: 0.6944 loss_val: 0.6918 euclidean dist: 0.5990 kl: 0.9420
Epoch:0042 loss_train: 0.6946 loss_val: 0.6909 euclidean dist: 0.5962 kl: 0.9396
Epoch:0043 loss_train: 0.6941 loss_val: 0.6913 euclidean dist: 0.5987 kl: 0.9422
Epoch:0044 loss_train: 0.6943 loss_val: 0.6905 euclidean dist: 0.5961 kl: 0.9387
Epoch:0045 loss_train: 0.6938 loss_val: 0.6908 euclidean dist: 0.5983 kl: 0.9419
Epoch:0046 loss_train: 0.6941 loss_val: 0.6898 euclidean dist: 0.5954 kl: 0.9404
Epoch:0047 loss_train: 0.6935 loss_val: 0.6904 euclidean dist: 0.5976 kl: 0.9428
Epoch:0048 loss_train: 0.6938 loss_val: 0.6902 euclidean dist: 0.5952 kl: 0.9396
Epoch:0049 loss_train: 0.6933 loss_val: 0.6897 euclidean dist: 0.5965 kl: 0.9446
Epoch:0050 loss_train: 0.6936 loss_val: 0.6890 euclidean dist: 0.5942 kl: 0.9443
Epoch:0051 loss_train: 0.6931 loss_val: 0.6893 euclidean dist: 0.5960 kl: 0.9475
Epoch:0052 loss_train: 0.6930 loss_val: 0.6885 euclidean dist: 0.5954 kl: 0.9465
Epoch:0053 loss_train: 0.6934 loss_val: 0.6889 euclidean dist: 0.5949 kl: 0.9414
Epoch:0054 loss_train: 0.6930 loss_val: 0.6877 euclidean dist: 0.5961 kl: 0.9484
Epoch:0055 loss_train: 0.6933 loss_val: 0.6883 euclidean dist: 0.5960 kl: 0.9408
Epoch:0056 loss_train: 0.6931 loss_val: 0.6867 euclidean dist: 0.5966 kl: 0.9481
Epoch:0057 loss_train: 0.6933 loss_val: 0.6874 euclidean dist: 0.5970 kl: 0.9423
Epoch:0058 loss_train: 0.6933 loss_val: 0.6857 euclidean dist: 0.5970 kl: 0.9475
Epoch:0059 loss_train: 0.6935 loss_val: 0.6863 euclidean dist: 0.5973 kl: 0.9420
Epoch:0060 loss_train: 0.6937 loss_val: 0.6851 euclidean dist: 0.5969 kl: 0.9448
Epoch:0061 loss_train: 0.6939 loss_val: 0.6849 euclidean dist: 0.5973 kl: 0.9420
Epoch:0062 loss_train: 0.6942 loss_val: 0.6841 euclidean dist: 0.5968 kl: 0.9426
Epoch:0063 loss_train: 0.6945 loss_val: 0.6835 euclidean dist: 0.5966 kl: 0.9420
Epoch:0064 loss_train: 0.6948 loss_val: 0.6829 euclidean dist: 0.5966 kl: 0.9410
Epoch:0065 loss_train: 0.6951 loss_val: 0.6825 euclidean dist: 0.5964 kl: 0.9403
Epoch:0066 loss_train: 0.6953 loss_val: 0.6820 euclidean dist: 0.5963 kl: 0.9392
Epoch:0067 loss_train: 0.6956 loss_val: 0.6816 euclidean dist: 0.5961 kl: 0.9383
Epoch:0068 loss_train: 0.6959 loss_val: 0.6814 euclidean dist: 0.5961 kl: 0.9374
Epoch:0069 loss_train: 0.6962 loss_val: 0.6811 euclidean dist: 0.5962 kl: 0.9380
Epoch:0070 loss_train: 0.6964 loss_val: 0.6810 euclidean dist: 0.5958 kl: 0.9383
Epoch:0071 loss_train: 0.6967 loss_val: 0.6807 euclidean dist: 0.5956 kl: 0.9396
Epoch:0072 loss_train: 0.6970 loss_val: 0.6806 euclidean dist: 0.5947 kl: 0.9399
Epoch:0073 loss_train: 0.6971 loss_val: 0.6802 euclidean dist: 0.5946 kl: 0.9404
Epoch:0074 loss_train: 0.6972 loss_val: 0.6802 euclidean dist: 0.5942 kl: 0.9393
Epoch:0075 loss_train: 0.6972 loss_val: 0.6798 euclidean dist: 0.5933 kl: 0.9418
Epoch:0076 loss_train: 0.6972 loss_val: 0.6799 euclidean dist: 0.5932 kl: 0.9404
Epoch:0077 loss_train: 0.6970 loss_val: 0.6799 euclidean dist: 0.5935 kl: 0.9414
Epoch:0078 loss_train: 0.6970 loss_val: 0.6799 euclidean dist: 0.5939 kl: 0.9422
Epoch:0079 loss_train: 0.6974 loss_val: 0.6800 euclidean dist: 0.5944 kl: 0.9460
Epoch:0080 loss_train: 0.6979 loss_val: 0.6799 euclidean dist: 0.5946 kl: 0.9513
Epoch:0081 loss_train: 0.6977 loss_val: 0.6800 euclidean dist: 0.5947 kl: 0.9453
Epoch:0082 loss_train: 0.6978 loss_val: 0.6800 euclidean dist: 0.5947 kl: 0.9420
Epoch:0083 loss_train: 0.6978 loss_val: 0.6800 euclidean dist: 0.5948 kl: 0.9393
Epoch:0084 loss_train: 0.6977 loss_val: 0.6801 euclidean dist: 0.5949 kl: 0.9373
Epoch:0085 loss_train: 0.6977 loss_val: 0.6801 euclidean dist: 0.5950 kl: 0.9361
Epoch:0086 loss_train: 0.6977 loss_val: 0.6802 euclidean dist: 0.5950 kl: 0.9352
Epoch:0087 loss_train: 0.6977 loss_val: 0.6803 euclidean dist: 0.5950 kl: 0.9348
Epoch:0088 loss_train: 0.6977 loss_val: 0.6804 euclidean dist: 0.5949 kl: 0.9346
Epoch:0089 loss_train: 0.6977 loss_val: 0.6804 euclidean dist: 0.5947 kl: 0.9347
Epoch:0090 loss_train: 0.6976 loss_val: 0.6803 euclidean dist: 0.5945 kl: 0.9351
Epoch:0091 loss_train: 0.6976 loss_val: 0.6805 euclidean dist: 0.5944 kl: 0.9359
Epoch:0092 loss_train: 0.6977 loss_val: 0.6809 euclidean dist: 0.5948 kl: 0.9365
Epoch:0093 loss_train: 0.6979 loss_val: 0.6811 euclidean dist: 0.5951 kl: 0.9334
Epoch:0094 loss_train: 0.6981 loss_val: 0.6810 euclidean dist: 0.5950 kl: 0.9295
Epoch:0095 loss_train: 0.6983 loss_val: 0.6811 euclidean dist: 0.5955 kl: 0.9281
Epoch:0096 loss_train: 0.6984 loss_val: 0.6810 euclidean dist: 0.5956 kl: 0.9279
Epoch:0097 loss_train: 0.6983 loss_val: 0.6807 euclidean dist: 0.5949 kl: 0.9314
Epoch:0098 loss_train: 0.6985 loss_val: 0.6811 euclidean dist: 0.5956 kl: 0.9347
Epoch:0099 loss_train: 0.6987 loss_val: 0.6809 euclidean dist: 0.5952 kl: 0.9324
Epoch:0100 loss_train: 0.6986 loss_val: 0.6815 euclidean dist: 0.5953 kl: 0.9320
Test:
 loss_test: 0.7118 euclidean dist: 0.6092 kl: 0.9708
end at: 2024-07-06 10:22:46.276599
elapsed time: 0.7916848166666667

"""




