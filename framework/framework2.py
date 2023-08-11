
from strategy_utils_framework.strategy import Strategy

class Framework2(Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args,strat_1,strat_2):
        super(Framework2, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
        self._X = X
        self._Y = Y
        self._X_te = X_te
        self._Y_te = Y_te
        self._idxs_lb = idxs_lb
        self._net = net
        self._handler = handler
        self._args = args
        self.strat_1=strat_1
        self.strat_2=strat_2


