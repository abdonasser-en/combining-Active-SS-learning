
from strategy_utils_framework.strategy import Strategy
import numpy as np 
import time 
from copy import deepcopy
class Framework2(Strategy):
    def __init__(self,X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(Framework2, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
        # self.X_tr = X
        # self.Y_tr = Y
        # self.X_te = X_te
        # self.Y_te = Y_te
        # self.idxs_lb = idxs_lb
        # self.model = net
        # self.net=deepcopy(net)
        # self.handler = handler
        # self.args = args

    # def train_framework(self,stratAl:object,stratSSL:object,NUM_ROUND,NUM_QUERY,alpha,n_epochs):

    #     print(f' Sratgey for active learning{stratAl} and strategy for semi-supervised learning used {stratSSL}')
    #     stratAl=stratAl(self.X_tr, self.Y_tr, self.X_te, self.Y_te, self.idxs_lb, self.net, self.handler, self.args,self.n_pool,self.device)
    #     stratSSL=stratSSL(self.X_tr, self.Y_tr, self.X_te, self.Y_te, self.idxs_lb, self.net, self.handler, self.args,self.n_pool,self.device,self.predict,self.g)


    #     self.train(alpha,n_epochs)

    #     test_acc=self.predict(self.X_te,self.Y_te)
    #     acc = np.zeros(NUM_ROUND+1)
    #     acc[0] = test_acc

    #     for rd in range(0, NUM_ROUND):
            
    #         if rd%2==0:
    #             # Al_methods
    #             print('Round {}/{}'.format(rd, NUM_ROUND), flush=True)
    #             labeled = len(np.arange(self.n_pool)[self.idxs_lb])
    #             if NUM_QUERY > int(self.args.nEnd*self.n_pool/100) - labeled:
    #                 NUM_QUERY = int(self.args.nEnd*self.n_pool/100) - labeled
                    
    #             # query
    #             ts = time.time()
    #             output = stratAl.query(NUM_QUERY)
    #             q_idxs = output
    #             self.idxs_lb[q_idxs] = True
    #             te = time.time()
    #             tp = te - ts
                
    #             # update
    #             self.update(self.idxs_lb)
    #             if hasattr(stratAl, 'train'):
                
    #                 best_test_acc=stratAl.train(alpha=2e-3, n_epoch=10)
    #             else: best_test_acc = self.train(alpha=2e-3, n_epoch=10)

    #             t_iter = time.time() - ts
                
    #             # round accuracy
    #             # test_acc = strategy.predict(X_te, Y_te)
    #             acc[rd] = best_test_acc
    #         else:
    #             #SSL methods
                
    #             print('Round {}/{}'.format(rd, NUM_ROUND), flush=True)
    #             labeled = len(np.arange(self.n_pool)[self.idxs_lb])
    #             if NUM_QUERY > int(self.args.nEnd*self.n_pool/100) - labeled:
    #                 NUM_QUERY = int(self.args.nEnd*self.n_pool/100) - labeled
                    
    #             # query
    #             ts = time.time()

    #             output = stratSSL.query(NUM_QUERY)
    #             q_idxs = output
    #             self.idxs_lb[q_idxs] = True
    #             te = time.time()
    #             tp = te - ts
                
    #             # update
    #             self.update(self.idxs_lb)
    #             best_test_acc = stratSSL.train(alpha=2e-3, n_epoch=10)

    #             t_iter = time.time() - ts
                
    #             # round accuracy
    #             # test_acc = strategy.predict(X_te, Y_te)
    #             acc[rd] = best_test_acc

    #     return acc