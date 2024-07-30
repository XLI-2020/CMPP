import numpy as np
import torch




def load_metr_la_data():
    A = np.load("data/MallB_adj.npy")
    X = np.load("data/MallB_five_mins.npy").transpose((1, 2, 0))
    print('A', A.shape, type(A))
    print('X', X.shape, type(X))
    return A, X


def eval_dist(out1, out2, label1, label2):
    delta_x = (out1 - label1)**2
    delta_y = (out2 - label2)**2
    delta_xy = delta_x + delta_y
    res = np.average(np.sqrt(delta_xy))
    return res

def eval_kl(u_q, v_q, u_p, v_p):
    """
    Parameters
    ----------
    P(x) ~ N(u_p, v_p) true data
    Q(x) ~ N(u_q, v_q) model output
    KL = P(x)Log(P(x)/Q(x))
    out1 : u_q
    out2 : v_q
    label1: u_p
    label2: v_p

    kl = 1/2(log(v_q**2) - log(v_p**2)) + (v_p + (u_p - u_q)**2)/(2*v_q)) - 1/2
    Returns
    -------
    """
    v_q_i = set(list(np.where(v_q>0)[0]))
    v_p_i = set(list(np.where(v_p>0)[0]))
    v_i = list(v_p_i.intersection(v_q_i))

    u_q = u_q[v_i]
    v_q = v_q[v_i]
    u_p = u_p[v_i]
    v_p = v_p[v_i]

    kl = 0.5*(np.log(v_q**2) - np.log(v_p**2)) + (v_p + (u_p - u_q)**2)/(2*v_q) - 0.5
    kl_avg = np.average(kl)
    return kl_avg


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []

    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, :, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)).double(), \
           torch.from_numpy(np.array(target)).double()
