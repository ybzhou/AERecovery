import numpy as np
from scipy.stats import entropy

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
    paired = set()
    for i in range(dist.shape[0]):
        idx = np.argsort(dist[i])[::-1]
        for j in idx:
            if (i, j) not in paired and (j, i) not in paired:
                paired.add((i,j))
                break

    ret = np.zeros(dist.shape[0])
    for i, j in paired:
        ret[i] = dist[i, j]
    return ret