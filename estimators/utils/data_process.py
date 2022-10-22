import copy
import numpy as np
def train_test_valid_split(features, labels, train_rate, validation_rate):
    features = np.array(features)
    labels = np.array(labels)
    idx = list(range(len(features)))
    idx_ordered = copy.copy(idx)
    idx_train = idx[:int(len(idx) * train_rate)]
    idx_validation = idx_ordered[int(len(idx)*train_rate):int(len(idx)*(train_rate+validation_rate))]
    idx_test = idx[int(len(idx) * (train_rate + validation_rate)):]
    np.random.shuffle(idx_train)
    print('len(idx_validation)', len(idx_validation))
    print('len(idx_test)', len(idx_test))
    return features, labels, idx_train, idx_validation, idx_test

def norm_Adj(W):
    assert W.shape[0] == W.shape[1]
    N = W.shape[0]
    D = np.diag(1/np.sqrt(np.sum(W, axis=1)))
    norm_Adj_matrix = np.dot(D, W)
    return norm_Adj_matrix


