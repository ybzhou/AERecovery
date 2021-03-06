import pickle
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.autograd import Variable
from ExpModules.AutoEncoder import AutoEncoder
from ExpModules.Utils import *


seed = 123
N = 50000
M = 200
D = 180
p = 0.02
l_max = 1
n_epoch = 1 #100
batch_size = 100
regc = torch.FloatTensor([0])
disp_freq = 100
cuda = torch.cuda.is_available()
torch.manual_seed(seed)
np.random.seed(seed)

model = AutoEncoder(D, M, F.relu) #, W.T)
if cuda:
    model.cuda()
    regc = Variable(regc.cuda())
else:
    regc = Variable(regc)

for noise_level in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]:
    h = BINS(N, M, p, l_max)
    # print(np.mean(h>0))
    randW = np.random.randn(M, D)
    W, _, _ = np.linalg.svd(randW, full_matrices=False)
    W = W/np.linalg.norm(W, 2, 1, True)
    data = np.dot(h, W)
    data += np.random.randn(*data.shape) * noise_level

    data -= data.mean(axis=0)

    #norm_data = torch.from_numpy(data).type(torch.FloatTensor)



    cost_func = nn.MSELoss(size_average=False)
    opt = optim.SGD(model.parameters(), lr = 0.01)
    # opt = optim.Adam(model.parameters(), lr = 1e-3)

    n_batch = N//batch_size
    cost = np.zeros(n_epoch*n_batch//disp_freq)
    coherence = np.zeros(n_epoch*n_batch//disp_freq)
    dot = np.zeros(n_epoch)
    dot_lower = np.zeros(n_epoch)
    dot_upper = np.zeros(n_epoch)
    apre = np.zeros(n_epoch)

    for epoch in range(n_epoch):
        running_loss = 0.0
        running_coherence = 0.0

        perm_idx = np.random.permutation(N)
        norm_data = torch.from_numpy(data[perm_idx]).type(torch.FloatTensor)
        if cuda:
            norm_data.cuda()

        h_recovered = None
        for batch_idx in range(n_batch):
            opt.zero_grad()

            if cuda:
                x = Variable(norm_data[batch_idx*batch_size: (batch_idx+1)*batch_size,:].cuda())
            else:
                x = Variable(norm_data[batch_idx*batch_size: (batch_idx+1)*batch_size,:])
            x_hat = model(x)
            if h_recovered is None:
                h_recovered = model.hidden().data.cpu().numpy()
            else:
                h_recovered = np.vstack((h_recovered, model.hidden().data.cpu().numpy()))

            loss = cost_func(x_hat, x) #+ regc*model.regularizer()
            loss.backward()
            opt.step()
            running_loss += loss.data[0]
            crt_W = model.W.data.cpu().numpy().astype('float64')
            crt_W = crt_W / np.linalg.norm(crt_W, 2, 0, True)
            dot_prod = np.dot(crt_W.T, crt_W) - np.eye(M)
            running_coherence += np.max(dot_prod)

            if batch_idx % disp_freq == disp_freq-1:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f, coherence: %.3f'  %
                      (epoch + 1, batch_idx + 1, running_loss / disp_freq,
                       running_coherence/disp_freq))
                # print( epoch*n_batch//disp_freq+batch_idx//disp_freq )
                cost[(epoch*n_batch+batch_idx)//disp_freq] = running_loss/disp_freq
                coherence[(epoch*n_batch+batch_idx)//disp_freq] = running_coherence/disp_freq
                running_loss = 0.0
                running_coherence = 0.0

        W_hat = model.W.data.cpu().numpy().astype('float64').T

        W_hat = W_hat / np.linalg.norm(W_hat, 2, 1, True)
        ret, pairs = greedy_pair(W, W_hat)
        # a = []
        # b = []
        # for i, j in enumerate(pairs):
        #     a.append(i)
        #     b.append(j)
        # truth = [i for i in range(M)]
        # print(sorted(b))
        # assert sorted(a) == truth, "aa"
        # assert sorted(b) == truth, "bb"
        dot[epoch] = np.percentile(ret, 50)
        dot_lower[epoch] = np.percentile(ret, 5)
        dot_upper[epoch] = np.percentile(ret, 95)
        print('dot med: {}, min: {}, max: {}'.format(dot[epoch], dot_lower[epoch], dot_upper[epoch]))
        apre[epoch] = APRE(h[perm_idx], h_recovered, p, pairs)
        print('APRE: {}'.format(apre[epoch]))

    # x_axis = np.linspace(0, n_epoch*batch_size, coherence.shape[0])
    # print(x_axis.shape, len(coherence))
    # fig, ax = plt.subplots()
    # l1, = ax.plot(x_axis, cost, '--', label='Training Cost')
    # plt.savefig("tr_cost.jpg")
    #
    # fig, ax = plt.subplots()
    # l2, = ax.plot(x_axis, coherence, '-', label='Coherence')
    # plt.savefig("coherence.jpg")
    #
    # fig, ax = plt.subplots()
    # l3 = ax.errorbar(range(n_epoch), dot, fmt='o', yerr=[dot-dot_lower, dot_upper-dot_lower], label='Dot Product')
    # plt.savefig("dot.jpg")
    # fig, ax = plt.subplots()
    # l4, = ax.plot(range(n_epoch), apre, label='APRE')
    # plt.savefig("apre.jpg")

    data = {'cost': cost,
            'coherence': coherence,
            'dot': (dot, dot_lower, dot_upper),
            'apre': apre}

    with open('results_n{0:.2f}_p{1:.2f}.pkl'.format(noise_level, p), 'wb') as f:
        pickle.dump(data, f)
    print('Done saving')
print('Done')

