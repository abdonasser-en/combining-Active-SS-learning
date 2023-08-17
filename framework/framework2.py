
from strategy_utils_framework.strategy import Strategy

class Framework2(Strategy):
    def __init__(self,X, Y, X_te, Y_te, idxs_lb, net, handler, args,stratAl,stratSSL):
        super(Framework2, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
        self.X_tr = X
        self.Y_tr = Y
        self.X_te = X_te
        self.Y_te = Y_te
        self.idxs_lb = idxs_lb
        self.model = net
        self.handler = handler
        self.args = args
        self.stratAl=stratAl(self.X_tr, self.Y_tr, self.X_te, self.Y_te, self.idxs_lb, self.net, self.handler, self.args,self.n_pool,self.device)
        self.stratSSL=stratSSL(self.X_tr, self.Y_tr, self.X_te, self.Y_te, self.idxs_lb, self.net, self.handler, self.args,self.n_pool,self.device,self.predict,self.g)
        


    


