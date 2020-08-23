import utils
import numpy as np
from acc_predictor.factory import get_acc_predictor


class AdaptiveSwitching:
    """ ensemble surrogate model """
    """ try all available models, pick one based on 10-fold crx vld """
    def __init__(self, n_fold=10):
        # self.model_pool = ['rbf', 'gp', 'mlp', 'carts']
        self.model_pool = ['rbf', 'gp', 'carts']
        self.n_fold = n_fold
        self.name = 'adaptive switching'
        self.model = None

    def fit(self, train_data, train_target):
        self._n_fold_validation(train_data, train_target, n=self.n_fold)

    def _n_fold_validation(self, train_data, train_target, n=10):

        n_samples = len(train_data)
        perm = np.random.permutation(n_samples)

        kendall_tau = np.full((n, len(self.model_pool)), np.nan)

        for i, tst_split in enumerate(np.array_split(perm, n)):
            trn_split = np.setdiff1d(perm, tst_split, assume_unique=True)

            # loop over all considered surrogate model in pool
            for j, model in enumerate(self.model_pool):

                acc_predictor = get_acc_predictor(model, train_data[trn_split], train_target[trn_split])

                rmse, rho, tau = utils.get_correlation(
                    acc_predictor.predict(train_data[tst_split]), train_target[tst_split])

                kendall_tau[i, j] = tau

        winner = int(np.argmax(np.mean(kendall_tau, axis=0) - np.std(kendall_tau, axis=0)))
        print("winner model = {}, tau = {}".format(self.model_pool[winner],
                                                   np.mean(kendall_tau, axis=0)[winner]))
        self.winner = self.model_pool[winner]
        # re-fit the winner model with entire data
        acc_predictor = get_acc_predictor(self.model_pool[winner], train_data, train_target)
        self.model = acc_predictor

    def predict(self, test_data):
        return self.model.predict(test_data)
