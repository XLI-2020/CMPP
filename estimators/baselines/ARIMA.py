from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from utils.data_process import train_test_valid_split
from utils.model_component import eval_kl, eval_dist
import torch


features = np.load('../input/MallB/five_mins/features_five.npy')
labels = np.load('../input/MallB/five_mins/labels_five.npy')
print(features.shape)
print(labels.shape)
features, labels, idx_train, idx_validation, idx_test = train_test_valid_split(features, labels, train_rate=0.7, validation_rate=0.1)

def label_transform(idx_train):
    label1, label2 = labels[idx_train][:, :, :, 0], labels[idx_train][:, :, :, 1]
    label1 = torch.reshape(label1, (-1, 1))
    label2 = torch.reshape(label2, (-1, 1))
    return label1, label2

out = []
label = []
features_ = features[:, :, :, 0]
labels_ = labels[:, :, :, 0]
nodes = features_.shape[2]


print('nodes', nodes)
print('labels_', labels_.shape)
pred_total = []
label_total = []

features_v = features[:, :, :, 1]
labels_v = labels[:, :, :, 1]
pred_total_v = []
label_total_v = []
for node in range(20,nodes):
    data = labels_[idx_train,:, node]
    data = np.reshape(data, (-1,1))
    test_data = labels_[idx_validation,:, node]
    test_data = np.reshape(test_data, (-1,1))
    pred_result = []
    label_result = []

    data_v = labels_v[idx_train, :, node]
    data_v = np.reshape(data_v, (-1, 1))
    test_data_v = labels_v[idx_validation, :, node]
    test_data_v = np.reshape(test_data_v, (-1, 1))
    pred_result_v = []
    label_result_v = []

    for i in range(len(idx_validation)):
        try:
            ###Mean prediction
            model = ARIMA(data, order=[5, 0, 0])
            ar_model = model.fit(disp=0)
            predict_ts = ar_model.predict()

            ###Var prediction
            model_v = ARIMA(data_v, order=[5,0,0])
            v_model = model_v.fit(disp=0)
            predict_v = v_model.predict()
            print('2222 predict_v', predict_v.shape)

            pred_result.append(predict_ts[0])
            label_result.append(test_data[i])
            data = np.concatenate([data, np.array(test_data[i]).reshape(-1,1)], axis=0)

            pred_result_v.append(predict_v[0])
            label_result_v.append(test_data_v[i])
            data_v = np.concatenate([data_v, np.array(test_data_v[i]).reshape(-1,1)], axis=0)
        except Exception as e:
            print(e)
            print('exception')
            pass

    pred_total.append(pred_result)
    label_total.append(label_result)

    pred_total_v.append(pred_result_v)
    label_total_v.append(label_result_v)

pred_total_m = np.array(pred_total).reshape(-1,1)
label_total_m = np.array(label_total).reshape(-1,1)
pred_total_v = np.array(pred_total_v).reshape(-1, 1)
label_total_v = np.array(pred_total_v).reshape(-1, 1)

eu_dist = eval_dist(pred_total_m, pred_total_v, label_total_m, label_total_v)
kl = eval_kl(pred_total_m, pred_total_v, label_total_m, label_total_v)


print('euclidean dist: {:.4f}'.format(eu_dist),
      'kl: {:.4f}'.format(kl))

