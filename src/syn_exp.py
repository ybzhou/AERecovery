import numpy as np
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
n_epoch = 100
batch_size = 100
regc = 1e-3
disp_freq = 100

torch.manual_seed(seed)
np.random.seed(seed)

h = BINS(N, M, p, l_max)
# print(np.mean(h>0))
randW = np.random.randn(M, D)
W, _, _ = np.linalg.svd(randW, full_matrices=False)
W = W/np.linalg.norm(W, 2, 1, True)
data = np.dot(h, W)

#data -= data.mean(axis=0)

norm_data = torch.from_numpy(data).type(torch.FloatTensor)


model = AutoEncoder(D, M, F.relu)

cost_func = nn.MSELoss(size_average=False)
# opt = optim.SGD(model.parameters(), lr = 0.01)
opt = optim.Adam(model.parameters(), lr = 1e-3)

cost = []
coherence = []
sparsity = []
dot = []
n_batch = N//batch_size
for epoch in range(n_epoch):
    running_loss = 0.0
    running_coherence = 0.0
    running_sparsity = 0.0
    h_recovered = None
    for batch_idx in range(n_batch):
        opt.zero_grad()

        x = Variable(norm_data[batch_idx*batch_size: (batch_idx+1)*batch_size,:])
        x_hat = model(x)
        if h_recovered is None:
            h_recovered = model.hidden().data.numpy()
        else:
            h_recovered = np.vstack((h_recovered, model.hidden().data.numpy()))

        loss = cost_func(x_hat, x) + regc*model.regularizer()
        loss.backward()
        opt.step()
        running_loss += loss.data[0]
        running_coherence += model.regularizer().data[0]
        running_sparsity += model.sparsity().data[0]/(batch_size*M)

        if batch_idx % disp_freq == disp_freq-1:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f, coherence: %.3f, sparsity: %.3f'  %
                  (epoch + 1, batch_idx + 1, running_loss / disp_freq, running_coherence/disp_freq,
                   running_sparsity/disp_freq))
            cost.append(running_loss/disp_freq)
            coherence.append(running_coherence/disp_freq)
            sparsity.append(running_sparsity/disp_freq)
            running_loss = 0.0
            running_coherence = 0.0
            running_sparsity = 0.0

    W_hat = model.W.data.numpy().astype('float64').T
    W_hat = W_hat / np.linalg.norm(W_hat, 2, 1, True)
    dot.append(greedy_pair(W, W_hat))
    print('avg dot: {}'.format(dot[-1]))


x_axis = np.linspace(0, n_epoch*batch_size, len(coherence))
print(x_axis.shape, len(coherence))
fig, ax = plt.subplots()
# l1, = ax.plot(cost, '--')
l2, = ax.plot(x_axis, coherence, '-')
# l3, = ax.plot(x_axis, sparsity)
l4, = ax.plot(x_axis[4::5], dot)
plt.savefig("result.jpg")
print('Done')

