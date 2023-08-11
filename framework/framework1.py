
class Strategy1:
    def method1(self, class_strategy1):
        return class_strategy1

class Strategy2:
    def method2(self, class_strategy2):
        return class_strategy2

class Framework1(Strategy1, Strategy2):
    def __init__(self, X_tr, Y_tr, X_te, Y_te, idxs_lb, model, handler, args, strategy1, strategy2):
        super().__init__()
        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.X_te = X_te
        self.Y_te = Y_te
        self.idxs_lb = idxs_lb
        self.model = model
        self.handler = handler
        self.args = args
        self.strategy1 = strategy1
        self.strategy2 = strategy2