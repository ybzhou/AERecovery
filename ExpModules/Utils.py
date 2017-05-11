import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

def BINS(n, d, p, l):
    prob = np.random.rand(n, d)
    y = np.random.rand(n, d)*l
    prob[prob < 1-p] = 0.0
    prob[prob >= 1-p] = y[prob >= 1-p]
    return prob


def KL_divergence(p_data, q_data, l_max, nbins=100):
    p_est, p_bin = np.histogram(p_data, bins=nbins,
                                range=(0, l_max), density=True)
    q_est, q_bin = np.histogram(q_data, bins=nbins,
                                range=(0, l_max), density=True)

    assert np.all(p_bin == q_bin), 'the bins should match'

    return entropy(p_est+1e-20, q_est+1e-20)


def greedy_pair(W, W_hat):
    dist = np.dot(W, W_hat.T)
    paired = []
    for i in range(dist.shape[0]):
        idx = np.argsort(dist[i])[::-1]
        for j in idx:
            if j not in paired:
                paired.append(j)
                break

    ret = np.zeros(dist.shape[0])
    assert len(paired) == dist.shape[0], "should have paired all vectors"
    for i, j in enumerate(paired):
        ret[i] = dist[i, j]
    return ret, paired


def APRE(h, h_hat, p, pairs, eps=0.1):
    w = np.zeros(h.shape)
    w[h>0] = 0.5/p
    w[h==0] = 0.5/(1-p)
    error = 0
    for i, j in enumerate(pairs):
        error += w[:,i] * (np.absolute(h[:,i]-h_hat[:,j])>eps)
    return error.sum()/(np.prod(h.shape))

def save_plot(x, y, color, label, x_label, y_label, save_file_name):
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color, label=label)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig(save_file_name)

def save_errorbar(x, y, y_lower, y_upper, color, label, x_label, y_label,
                  save_file_name):
    fig, ax = plt.subplots()
    ax.errorbar(x, y, color=color, yerr=[y-y_lower, y_upper-y],
                 label=label, elinewidth=1, capsize=3)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig(save_file_name)