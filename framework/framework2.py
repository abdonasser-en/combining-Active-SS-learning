
from strategy_utils_framework.strategy import Strategy

class Framework2(Strategy):
    def __init__(self,X, Y, X_te, Y_te, idxs_lb, net, handler, args,active_strategy, semi_supervised_strategy):
        super(Framework2, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
        self.active_strategy=active_strategy
        self.strat2=semi_supervised_strategy
    


