import numpy as np
from utils.data_process import train_test_valid_split,norm_Adj
from utils.model_component import ParalleStGcn
import torch
import torch.nn as nn
import math
import time
from torch import optim
from datetime import datetime
from utils.model_component import SaGcnGru,Gru, integrated_loss,ParalleStGcn, eval_kl, eval_dist
np.random.seed(2021)
st = datetime.now()
print('start at:', st)
for t in ['one_min', 'five_mins', 'ten_mins']:

    print('!!!!frequency:', t)
    features = np.load('./input/MallA/{t1}/features_{t2}.npy'.format(t1=t, t2=t.split('_')[0]))
    labels = np.load('./input/MallA/{t1}/labels_{t2}.npy'.format(t1=t, t2=t.split('_')[0]))
    print(features.shape)
    print(labels.shape)

    # features = np.load('./input/MallA/ten_mins/features_ten.npy')
    # labels = np.load('./input/MallA/ten_mins/labels_ten.npy')
    # print(features.shape)
    # print(labels.shape)


    features, labels, idx_train, idx_validation, idx_test = train_test_valid_split(features, labels, train_rate=0.7, validation_rate=0.1)
    adj = np.load('./input/MallA/Adj.npy')
    print('Adj shape',adj.shape)
    norm_adj = norm_Adj(adj)

    features = torch.FloatTensor(features)
    features = torch.transpose(features,1,2)
    labels = torch.FloatTensor(labels)
    labels = torch.transpose(labels,1,2)
    # labels shape (2530, 125, 1, 2)

    norm_adj = torch.FloatTensor(norm_adj)
    in_channels = 2
    out_channels = 16
    hidden_size = 16
    output_size = 1
    LR = 0.01

    time_interval = 'ten_secs'#1min, 5mins
    model = ParalleStGcn(norm_adj,in_channels,out_channels,hidden_size,output_size)
    # estimators = Gru(in_channels,hidden_size,output_size)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    batch_size = 64
    batches = int(len(idx_train)/batch_size)


    def label_transform(idx_train):
        label1, label2 = labels[idx_train][:, :, :, 0], labels[idx_train][:, :, :, 1]
        label1 = torch.reshape(label1, (-1, output_size))
        label2 = torch.reshape(label2, (-1, output_size))
        return label1, label2

    # def trans_output(out_mean, out_var):
    #     # print('columns', len(columns))
    #     out_mean = out_mean.reshape(-1, len(columns)) #b*n
    #     # print('out_mean', out_mean.shape)
    #     out_var = out_var.reshape(-1, len(columns))
    #     out_idx = torch.unsqueeze(torch.tensor(idx_validation),dim=-1)
    #     # print('out_idx shape',out_idx.shape)
    #     result = []
    #     for i,c in enumerate(columns):
    #         out_c = torch.cat([out_idx,out_mean[:,[i]], out_var[:,[i]]],dim=-1)
    #         out_c_array = out_c.detach().numpy()
    #         # print(out_c_array.shape)
    #         out_c_list = list(map(lambda x: ','.join(list(map(lambda y:str(y),x))),list(out_c_array)))
    #         # print('out_c_list',out_c_list)
    #         out_c_list.insert(0,str(c))
    #         # print('out_c_list len', len(out_c_list))
    #         result.append(out_c_list)
    #     return result

    Memory_usage_list =  []
    def train(epoch):
        t = time.time()
        model.train()
        for batch in range(batches):
            optimizer.zero_grad()
            batch_idx = idx_train[batch_size*batch:batch_size*(batch+1)]
            out1, out2 = model(features[batch_idx])
            label1, label2 = label_transform(batch_idx)
            loss = integrated_loss(out1,out2,label1,label2)
            loss.backward()
            optimizer.step()
        model.eval()

        vl = len(idx_validation)
        vst = time.time()
        # pid = os.getpid()
        # py = psutil.Process(pid)
        # memoryUse_st = py.memory_info()[0] / 2. ** 10  # originally memory use in GB...I think
        out_val1, out_val2 = model(features[idx_validation])
        # memoryUse_et = py.memory_info()[0] / 2. ** 10
        # print('memory total used for this operation:', round((memoryUse_et - memoryUse_st)/1024, 1))
        # Memory_usage_list.append(round((memoryUse_et - memoryUse_st)/1024, 1))
        # print('average memory usage:', round(sum(Memory_usage_list)/len(Memory_usage_list), 1))
        # print('memory  used per sample for this operation:', (memoryUse_et - memoryUse_st)/vl) #473kb
        # print('memory use before counting:', memoryUse_st)
        # print('memory use after counting:', memoryUse_et)
        vet = time.time()
        # print('validation spent time: ', vet-vst)
        # print('out_val1 shape',out_val1.shape)
        label_val1, label_val2 = label_transform(idx_validation)
        loss_val = integrated_loss(out_val1, out_val2, label_val1, label_val2)
        # loss_val = F.mse_loss(out_val1,label_val1)
        # loss_val = F.mse_loss(out_val2,label_val2)
        out_val = torch.cat([out_val1,out_val2],dim=0)
        label_val = torch.cat([label_val1,label_val2],dim=0)
        # query_pre = trans_output(out_val1, out_val2)
        # query_true = trans_output(label_val1,label_val2)
        # if not os.path.exists(f'./query_data/{time_interval}/{epoch}'):
        #     os.makedirs(f'./query_data/{time_interval}/{epoch}')
        # query_pre_df = pd.DataFrame(query_pre)
        # query_true_df = pd.DataFrame(query_true)
        # query_pre_df.to_csv(f'./query_data/{time_interval}/{epoch}/query_pre.csv',sep='\t',header=False, index=False)
        # query_true_df.to_csv(f'./query_data/{time_interval}/{epoch}/query_true.csv',sep='\t',header=False, index=False)

        eu_dist = eval_dist(out_val1.detach().numpy(), out_val2.detach().numpy(), label_val1.detach().numpy(), label_val2.detach().numpy())
        kl = eval_kl(out_val1.detach().numpy(), out_val2.detach().numpy(), label_val1.detach().numpy(), label_val2.detach().numpy())

        # rmse1, mae1, acc1 = evaluation_(label_val1.detach().numpy(),out_val1.detach().numpy())
        # acc1_mape = evaluation(label_val1.detach().numpy(),out_val1.detach().numpy())[-1]
        # rmse2, mae2, acc2 = evaluation(label_val2.detach().numpy(),out_val2.detach().numpy())
        # acc2_norm = evaluation_(label_val2.detach().numpy(),out_val2.detach().numpy())[-1]
        # rmse, mae, acc = evaluation(label_val.detach().numpy(),out_val.detach().numpy())

        # print('eu',type(eu_dist), eu_dist)
        # print('kl',type(kl), kl)
        print('Epoch:{:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'euclidean dist: {:.4f}'.format(eu_dist),
              'kl: {:.4f}'.format(kl))
        # print('Epoch: {:04d}'.format(epoch+1),
        #       'loss_train: {:.4f}'.format(loss.item()),
        #       'loss_val: {:.4f}'.format(loss_val.item()),
        #       'acc_val_MEAN(mape): {:.4f}'.format(acc1_mape),
        #       'acc_val_MEAN(norm): {:.4f}'.format(acc1),
        #       'mae_val_MEAN{:.4f}'.format(mae1),
        #       'rmse_val_MEAN:{:.4f}'.format(rmse1),
        #       'acc_val_VAR(mape): {:.4f}'.format(acc2),
        #       'acc_val_VAR(norm): {:.4f}'.format(acc2_norm),
        #       'mae_val_VAR:{:.4f}'.format(mae2),
        #       'rmse_val_VAR:{:.4f}'.format(rmse2),
        #       'time: {:.4f}s'.format(time.time() - t),
        #       'validation dataset len:{vl}'.format(vl = vl),
        #       'validation spent time: {tl}'.format(tl=vet-vst),
        #       'validation time per sample:{pl}'.format(pl=round((vet-vst)/vl, 3)))

        return eu_dist, kl
        # return loss.item(),loss_val.item(),acc1,acc1_mape,mae1,rmse1,acc2_norm,acc2,mae2,rmse2,(time.time() - t), round((vet-vst)/vl, 3)

    def test():
        model.eval()
        out_val1, out_val2 = model(features[idx_test])
        label_val1, label_val2 = label_transform(idx_test)
        loss_test = integrated_loss(out_val1, out_val2, label_val1, label_val2)
        eu_dist = eval_dist(out_val1.detach().numpy(), out_val2.detach().numpy(), label_val1.detach().numpy(),
                            label_val2.detach().numpy())
        kl = eval_kl(out_val1.detach().numpy(), out_val2.detach().numpy(), label_val1.detach().numpy(),
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
        e_dist, kl =  train(epoch)
        # loss_train, loss_val, acc_val_1, acc_val_1_mape, mae_val_1, rmse_val_1,acc_val_2,acc_val_2_mape,mae_val_2,rmse_val_2, elasped_time, etps = train(epoch)
        # time_eva.append(etps)
        # epoch_results.append([loss_train, loss_val, acc_val_1, acc_val_1_mape, mae_val_1, rmse_val_1,acc_val_2,acc_val_2_mape,mae_val_2,rmse_val_2, elasped_time])
        # print('average epts among epoch:', np.average(time_eva))
        # break

    test()

et = datetime.now()

print('end at:', et)

dt = (et - st).total_seconds() / 3600  # hours
print('elapsed time:', str(dt))
    # best_performance_result = sorted(epoch_results,key=lambda x:x[3]+x[7],reverse=True)[0]
    # cols = ['loss_train','loss_val','acc_val_norm','acc_val_mape','mae_val','rmse_val','time']
    # cols = ['loss_train','loss_val','acc_val_MEAN(norm)','acc_val_MEAN(mape)','mae_val_MEAN','rmse_val_MEAN','acc_val_VAR(norm)','acc_val_VAR(mape)','mae_val_VAR','rmse_val_VAR','time']
    # best_performance_result = dict(list(zip(cols,best_performance_result)))
    # print('best_performance_result\n', best_performance_result)
    # res_df = pd.DataFrame(epoch_results)
    # res_df.columns = cols
    # res_df.to_csv('./results/ten_secs_sa_gcn_gru_variance_mape.csv',header=True)

    # optimizer = optim.Adam(estimators.parameters(),lr=LR*0.1,weight_decay=0)
    # for epoch in range(fine_train_epochs):
    #     train(epoch)

    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # def test():
    #     t = time.time()
    #     estimators.eval()
    #
    #     out_test1, out_test2 = estimators(features[idx_test])
    #     label_test1, label_test2 = label_transform(idx_test)
    #     loss_test = integrated_loss(out_test1, out_test2, label_test1, label_test2)
    #
    #     rmse1, mae1, acc1 = evaluation_(label_test1.detach().numpy(), out_test1.detach().numpy())
    #     acc1_mape = evaluation(label_test1.detach().numpy(), out_test1.detach().numpy())[-1]
    #     rmse2, mae2, acc2 = evaluation(label_test2.detach().numpy(), out_test2.detach().numpy())
    #     acc2_norm = evaluation_(label_test2.detach().numpy(), out_test2.detach().numpy())[-1]
    #     print("Test set results:",
    #           'loss_test: {:.4f}'.format(loss_test.item()),
    #           'acc_test_MEAN(mape): {:.4f}'.format(acc1_mape),
    #           'acc_test_MEAN(norm): {:.4f}'.format(acc1),
    #           'mae_test_MEAN{:.4f}'.format(mae1),
    #           'rmse_test_MEAN:{:.4f}'.format(rmse1),
    #           'acc_test_VAR(mape): {:.4f}'.format(acc2),
    #           'acc_test_VAR(norm): {:.4f}'.format(acc2_norm),
    #           'mae_test_VAR:{:.4f}'.format(mae2),
    #           'rmse_test_VAR:{:.4f}'.format(rmse2),
    #           'time: {:.4f}s'.format(time.time() - t))
    #
    # test()


