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