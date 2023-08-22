import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pdb
class LeastConfidence:
    def __init__(self, X, Y,  X_te, Y_te, idxs_lb, net, handler, args,n_pool,device):
        #super(LeastConfidence, self).__init__(X, Y, X_te, Y_te,  idxs_lb, net, handler, args)
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
        
    def predict_prob(self, X, Y):
        transform = self.args.transform_te 
        loader_te = DataLoader(self.handler(X, Y, 
                        transform=transform), shuffle=False, pin_memory=True, **self.args.loader_te_args)

        self.model.eval()

        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data
        
        return probs

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X_tr[idxs_unlabeled], np.asarray(self.Y_tr)[idxs_unlabeled])
        U = probs.max(1)[0]
        return idxs_unlabeled[U.sort()[1][:n]]
