import os
import json
import argparse
import numpy as np
from pymoo.factory import get_decomposition
from pymoo.visualization.scatter import Scatter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.model.decision_making import DecisionMaking, normalize, find_outliers_upper_tail, NeighborFinder

_DEBUG = False


class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, n_survive=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected

    def _do(self, F, **kwargs):
        n, m = F.shape

        if self.normalize:
            F = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True)

        neighbors_finder = NeighborFinder(F, epsilon=0.125, n_min_neigbors="auto", consider_2d=False)

        mu = np.full(n, - np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):

            # for each neighbour in a specific radius of that solution
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)
        if self.n_survive is not None:
            return np.argsort(mu)[-self.n_survive:]
        else:
            return find_outliers_upper_tail(mu)  # return points with trade-off > 2*sigma


def main(args):
    # preferences
    if args.prefer is not None:
        preferences = {}
        for p in args.prefer.split("+"):
            k, v = p.split("#")
            if k == 'top1':
                preferences[k] = 100 - float(v)  # assuming top-1 accuracy
            else:
                preferences[k] = float(v)
        weights = np.fromiter(preferences.values(), dtype=float)

    archive = json.load(open(args.expr))['archive']
    subnets, top1, sec_obj = [v[0] for v in archive], [v[1] for v in archive], [v[2] for v in archive]
    sort_idx = np.argsort(top1)
    F = np.column_stack((top1, sec_obj))[sort_idx, :]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    pf = F[front, :]
    ps = np.array(subnets)[sort_idx][front]

    if args.prefer is not None:
        # choose the architectures thats closest to the preferences
        I = get_decomposition("asf").do(pf, weights).argsort()[:args.n]
    else:
        # choose the architectures with highest trade-off
        dm = HighTradeoffPoints(n_survive=args.n)
        I = dm.do(pf)

    # always add most accurate architectures
    I = np.append(I, 0)

    # create the supernet
    from evaluator import OFAEvaluator
    supernet = OFAEvaluator(model_path=args.supernet_path)

    for idx in I:
        save = os.path.join(args.save, "net-flops@{:.0f}".format(pf[idx, 1]))
        os.makedirs(save, exist_ok=True)
        subnet, _ = supernet.sample({'ks': ps[idx]['ks'], 'e': ps[idx]['e'], 'd': ps[idx]['d']})
        with open(os.path.join(save, "net.subnet"), 'w') as handle:
            json.dump(ps[idx], handle)
        supernet.save_net_config(save, subnet, "net.config")
        supernet.save_net(save, subnet, "net.inherited")

    if _DEBUG:
        print(ps[I])
        plot = Scatter()
        plot.add(pf, alpha=0.2)
        plot.add(pf[I, :], color="red", s=100)
        plot.show()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    parser.add_argument('--expr', type=str, default='',
                        help='location of search experiment dir')
    parser.add_argument('--prefer', type=str, default=None,
                        help='preferences in choosing architectures (top1#80+flops#150)')
    parser.add_argument('-n', type=int, default=1,
                        help='number of architectures desired')
    parser.add_argument('--supernet_path', type=str, default='./data/ofa_mbv3_d234_e346_k357_w1.0',
                        help='file path to supernet weights')

    cfgs = parser.parse_args()
    main(cfgs)
