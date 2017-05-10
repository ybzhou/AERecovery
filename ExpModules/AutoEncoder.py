import torch
import torch.nn as nn
import torch.nn.init as init

from torch.autograd import Variable
from torch.nn.parameter import Parameter

class AutoEncoder(nn.Module):
    def __init__(self, x_dim, h_dim, act_func):
        super(AutoEncoder, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.f = act_func
        self.W = Parameter(torch.FloatTensor(x_dim, h_dim))
        self.bn = nn.BatchNorm1d(h_dim)
        init.orthogonal(self.W)
        # init.xavier_uniform(self.W)

    def forward(self, x):
        W_norm = torch.norm(self.W, p=2, dim=0)
        W_star = self.W / W_norm.expand_as(self.W)
        self.h = self.f(self.bn(torch.mm(x, W_star)))
        x_hat = torch.mm(self.h, W_star.t())
        return x_hat

    def regularizer(self):
        W_norm = torch.norm(self.W, p=2, dim=0)
        W_star = self.W / W_norm.expand_as(self.W)
        coherence = torch.abs(W_star.t().dot(W_star))
        reg = torch.sum(coherence)
        return reg

    def sparsity(self):
        return torch.gt(self.h, 0).sum()

    def hidden(self):
        return self.h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.x_dim) + ' -> ' \
            + str(self.h_dim) + ' -> ' \
            + str(self.x_dim) + ')'