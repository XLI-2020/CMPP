#NB: In order to fit into our setting, the code here is adapted from: https://github.com/FelixOpolka/STGCN-PyTorch
import argparse
import torch
import torch.nn as nn
from stgcn import STGCN
from utils import generate_dataset, load_metr_la_data, get_normalized_adj,eval_dist,eval_kl
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


num_timesteps_input = 12
num_timesteps_output = 3

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--time_interval', type=str, default='five_mins')
parser.add_argument('--device', type=str,
                    default='cpu')

args = parser.parse_args()
epochs = args.epochs
batch_size = args.batch_size
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    net.train()
    for i in range(0, training_input.shape[0], batch_size):
        optimizer.zero_grad()

        print('i',i)
        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        # print('training', 'A', A_wave.shape, X_batch.shape)
        out_m, out_v = net(A_wave, X_batch)
        y_batch_m = y_batch[:,:,0,:]
        y_batch_v = y_batch[:,:,1,:]
        loss_m = loss_criterion(out_m, y_batch_m)
        loss_v = loss_criterion(out_v, y_batch_v)
        loss = loss_m + loss_v

        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    torch.manual_seed(7)
    A, X = load_metr_la_data()
    split_line1 = int(X.shape[2] * 0.7)
    split_line2 = int(X.shape[2] * 0.8)
    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input_total, val_target_total = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)
    print('training_input', training_input.shape)#training_input torch.Size([328, 207, 12, 2])

    print('training_target', training_target.shape) #training_target torch.Size([328, 207, 3])

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave.to(device=args.device)

    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)

    net = net.double()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    e_dists = []
    e_kls = []
    for epoch in range(epochs):
        st = time.time()
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        elapsed = time.time()-st
        print('training time:', elapsed)
        training_losses.append(loss)
    print("Training loss: {}".format(training_losses[-1]))
    net.train(False)
    with torch.no_grad():
        for i in range(0, val_input_total.shape[0], batch_size):
            val_input, val_target = val_input_total[i:i+batch_size,:,:,:], val_target_total[i:i+batch_size,:,:,:]
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)
            vl = val_input.shape[0]
            out_m, out_v = net(A_wave, val_input)
            out_m = torch.reshape(out_m, (-1,1))
            out_v = torch.reshape(out_v, (-1,1))

            val_tar_m = val_target[:, :, 0, :]
            val_tar_m = torch.reshape(val_tar_m, (-1,1))

            val_tar_v = val_target[:, :, 1, :]
            val_tar_v = torch.reshape(val_tar_v,(-1,1))

            loss_m = loss_criterion(out_m, val_tar_m)
            loss_v = loss_criterion(out_v, val_tar_v)
            val_loss = loss_m + loss_v
            validation_losses.append(val_loss)
            e_dist = eval_dist(out_m.cpu().numpy(), out_v.cpu().numpy(), val_tar_m.cpu().numpy(), val_tar_v.cpu().numpy())
            e_dists.append(e_dist)
            kl = eval_kl(out_m.cpu().numpy(), out_v.cpu().numpy(), val_tar_m.cpu().numpy(), val_tar_v.cpu().numpy())
            e_kls.append(kl)
        print('val loss', sum(validation_losses)/len(validation_losses))
        print('val data e_dist:', sum(e_dists)/len(e_dists))
        print('val data kl:', sum(e_kls)/len(e_kls))


