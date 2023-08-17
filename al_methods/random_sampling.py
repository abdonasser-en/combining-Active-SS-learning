import numpy as np
from strategy_utils_framework.strategy import Strategy
import pdb

class RandomSampling:
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args,n_pool,device):
        # super(RandomSampling, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
        self.X_tr = X
        self.Y_tr = Y
        self.X_te = X_te
        self.Y_te = Y_te
        self.idxs_lb = idxs_lb
        self.model = net
        self.handler = handler
        self.args = args
        self.n_pool=n_pool
        self.device=device

def query(self, n):
    inds = np.where(self.idxs_lb==0)[0]
    return inds[np.random.permutation(len(inds))][:n]

