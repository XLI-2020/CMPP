from sklearn.svm import SVR
import numpy as np
from utils.data_process import train_test_valid_split
from utils.model_component import eval_kl, eval_dist


features = np.load('../input/MallB/five_mins/features_five.npy')
labels = np.load('../input/MallB/five_mins/labels_five.npy')



features, labels, idx_train, idx_validation, idx_test = train_test_valid_split(features, labels, train_rate=0.7, validation_rate=0.1)



features_m = np.transpose(np.array(features[:,:,100:110,0]),axes=(0,2,1))

features_v = np.transpose(np.array(features[:,:,100:110,1]),axes=(0,2,1))

labels_m = np.transpose(np.array(labels[:,:,100:110,0]),axes=(0,2,1))
labels_v = np.transpose(np.array(labels[:,:,100:110,1]),axes=(0,2,1))

train_m = np.reshape(features_m[idx_train,:,:],(-1,10))
test_m =  np.reshape(features_m[idx_validation,:,:],(-1,10))

train_v = np.reshape(features_v[idx_train,:,:],(-1,10))
test_v =  np.reshape(features_v[idx_validation,:,:],(-1,10))

train_label_m = np.reshape(labels_m[idx_train,:,:],(-1,1))
test_label_m = np.reshape(labels_m[idx_validation,:,:],(-1,1))

train_label_v = np.reshape(labels_v[idx_train,:,:],(-1,1))
test_label_v = np.reshape(labels_v[idx_validation,:,:],(-1,1))

svr_m = SVR(kernel='rbf')

svr_m.fit(train_m, train_label_m)

pred_m = svr_m.predict(test_m)

svr_v = SVR()
svr_v.fit(train_v, train_label_v)
pred_v = svr_v.predict(test_v)


e_dist = eval_dist(pred_m, pred_v, test_label_m, test_label_v)
kl = eval_kl(pred_m, pred_v, test_label_m, test_label_v)

print('e_dist, kl:', e_dist, kl)

