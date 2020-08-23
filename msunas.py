import os
import json
import shutil
import argparse
import subprocess
import numpy as np
from utils import get_correlation
from evaluator import OFAEvaluator, get_net_info

from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.factory import get_performance_indicator
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.factory import get_algorithm, get_crossover, get_mutation

from search_space.ofa import OFASearchSpace
from acc_predictor.factory import get_acc_predictor
from utils import prepare_eval_folder, MySampling, BinaryCrossover, MyMutation

_DEBUG = False
if _DEBUG: from pymoo.visualization.scatter import Scatter


class MSuNAS:
    def __init__(self, kwargs):
        self.search_space = OFASearchSpace()
        self.save_path = kwargs.pop('save', '.tmp')  # path to save results
        self.resume = kwargs.pop('resume', None)  # resume search from a checkpoint
        self.sec_obj = kwargs.pop('sec_obj', 'flops')  # second objective to optimize simultaneously
        self.iterations = kwargs.pop('iterations', 30)  # number of iterations to run search
        self.n_doe = kwargs.pop('n_doe', 100)  # number of architectures to train before fit surrogate model
        self.n_iter = kwargs.pop('n_iter', 8)  # number of architectures to train in each iteration
        self.predictor = kwargs.pop('predictor', 'rbf')  # which surrogate model to fit
        self.n_gpus = kwargs.pop('n_gpus', 1)  # number of available gpus
        self.gpu = kwargs.pop('gpu', 1)  # required number of gpus per evaluation job
        self.data = kwargs.pop('data', '../data')  # location of the data files
        self.dataset = kwargs.pop('dataset', 'imagenet')  # which dataset to run search on
        self.n_classes = kwargs.pop('n_classes', 1000)  # number of classes of the given dataset
        self.n_workers = kwargs.pop('n_workers', 6)  # number of threads for dataloader
        self.vld_size = kwargs.pop('vld_size', 10000)  # number of images from train set to validate performance
        self.trn_batch_size = kwargs.pop('trn_batch_size', 96)  # batch size for SGD training
        self.vld_batch_size = kwargs.pop('vld_batch_size', 250)  # batch size for validation
        self.n_epochs = kwargs.pop('n_epochs', 5)  # number of epochs to SGD training
        self.test = kwargs.pop('test', False)  # evaluate performance on test set
        self.supernet_path = kwargs.pop(
            'supernet_path', './data/ofa_mbv3_d234_e346_k357_w1.0')  # supernet model path
        self.latency = self.sec_obj if "cpu" in self.sec_obj or "gpu" in self.sec_obj else None

    def search(self):

        if self.resume:
            archive = self._resume_from_dir()
        else:
            # the following lines corresponding to Algo 1 line 1-7 in the paper
            archive = []  # initialize an empty archive to store all trained CNNs

            # Design Of Experiment
            if self.iterations < 1:
                arch_doe = self.search_space.sample(self.n_doe)
            else:
                arch_doe = self.search_space.initialize(self.n_doe)

            # parallel evaluation of arch_doe
            top1_err, complexity = self._evaluate(arch_doe, it=0)

            # store evaluated / trained architectures
            for member in zip(arch_doe, top1_err, complexity):
                archive.append(member)

        # reference point (nadir point) for calculating hypervolume
        ref_pt = np.array([np.max([x[1] for x in archive]), np.max([x[2] for x in archive])])

        # main loop of the search
        for it in range(1, self.iterations + 1):

            # construct accuracy predictor surrogate model from archive
            # Algo 1 line 9 / Fig. 3(a) in the paper
            acc_predictor, a_top1_err_pred = self._fit_acc_predictor(archive)

            # search for the next set of candidates for high-fidelity evaluation (lower level)
            # Algo 1 line 10-11 / Fig. 3(b)-(d) in the paper
            candidates, c_top1_err_pred = self._next(archive, acc_predictor, self.n_iter)

            # high-fidelity evaluation (lower level)
            # Algo 1 line 13-14 / Fig. 3(e) in the paper
            c_top1_err, complexity = self._evaluate(candidates, it=it)

            # check for accuracy predictor's performance
            rmse, rho, tau = get_correlation(
                np.vstack((a_top1_err_pred, c_top1_err_pred)), np.array([x[1] for x in archive] + c_top1_err))

            # add to archive
            # Algo 1 line 15 / Fig. 3(e) in the paper
            for member in zip(candidates, c_top1_err, complexity):
                archive.append(member)

            # calculate hypervolume
            hv = self._calc_hv(
                ref_pt, np.column_stack(([x[1] for x in archive], [x[2] for x in archive])))

            # print iteration-wise statistics
            print("Iter {}: hv = {:.2f}".format(it, hv))
            print("fitting {}: RMSE = {:.4f}, Spearman's Rho = {:.4f}, Kendall’s Tau = {:.4f}".format(
                self.predictor, rmse, rho, tau))

            # dump the statistics
            with open(os.path.join(self.save_path, "iter_{}.stats".format(it)), "w") as handle:
                json.dump({'archive': archive, 'candidates': archive[-self.n_iter:], 'hv': hv,
                           'surrogate': {
                               'model': self.predictor, 'name': acc_predictor.name,
                               'winner': acc_predictor.winner if self.predictor == 'as' else acc_predictor.name,
                               'rmse': rmse, 'rho': rho, 'tau': tau}}, handle)
            if _DEBUG:
                # plot
                plot = Scatter(legend={'loc': 'lower right'})
                F = np.full((len(archive), 2), np.nan)
                F[:, 0] = np.array([x[2] for x in archive])  # second obj. (complexity)
                F[:, 1] = 100 - np.array([x[1] for x in archive])  # top-1 accuracy
                plot.add(F, s=15, facecolors='none', edgecolors='b', label='archive')
                F = np.full((len(candidates), 2), np.nan)
                F[:, 0] = np.array(complexity)
                F[:, 1] = 100 - np.array(c_top1_err)
                plot.add(F, s=30, color='r', label='candidates evaluated')
                F = np.full((len(candidates), 2), np.nan)
                F[:, 0] = np.array(complexity)
                F[:, 1] = 100 - c_top1_err_pred[:, 0]
                plot.add(F, s=20, facecolors='none', edgecolors='g', label='candidates predicted')
                plot.save(os.path.join(self.save_path, 'iter_{}.png'.format(it)))

        return

    def _resume_from_dir(self):
        """ resume search from a previous iteration """
        import glob

        archive = []
        for file in glob.glob(os.path.join(self.resume, "net_*.subnet")):
            arch = json.load(open(file))
            pre, ext = os.path.splitext(file)
            stats = json.load(open(pre + ".stats"))
            archive.append((arch, 100 - stats['top1'], stats[self.sec_obj]))

        return archive

    def _evaluate(self, archs, it):
        gen_dir = os.path.join(self.save_path, "iter_{}".format(it))
        prepare_eval_folder(
            gen_dir, archs, self.gpu, self.n_gpus, data=self.data, dataset=self.dataset,
            n_classes=self.n_classes, supernet_path=self.supernet_path,
            num_workers=self.n_workers, valid_size=self.vld_size,
            trn_batch_size=self.trn_batch_size, vld_batch_size=self.vld_batch_size,
            n_epochs=self.n_epochs, test=self.test, latency=self.latency, verbose=False)

        subprocess.call("sh {}/run_bash.sh".format(gen_dir), shell=True)

        top1_err, complexity = [], []

        for i in range(len(archs)):
            try:
                stats = json.load(open(os.path.join(gen_dir, "net_{}.stats".format(i))))
            except FileNotFoundError:
                # just in case the subprocess evaluation failed
                stats = {'top1': 0, self.sec_obj: 1000}  # makes the solution artificially bad so it won't survive
                # store this architecture to a separate in case we want to revisit after the search
                os.makedirs(os.path.join(self.save_path, "failed"), exist_ok=True)
                shutil.copy(os.path.join(gen_dir, "net_{}.subnet".format(i)),
                            os.path.join(self.save_path, "failed", "it_{}_net_{}".format(it, i)))

            top1_err.append(100 - stats['top1'])
            complexity.append(stats[self.sec_obj])

        return top1_err, complexity

    def _fit_acc_predictor(self, archive):
        inputs = np.array([self.search_space.encode(x[0]) for x in archive])
        targets = np.array([x[1] for x in archive])
        assert len(inputs) > len(inputs[0]), "# of training samples have to be > # of dimensions"

        acc_predictor = get_acc_predictor(self.predictor, inputs, targets)

        return acc_predictor, acc_predictor.predict(inputs)

    def _next(self, archive, predictor, K):
        """ searching for next K candidate for high-fidelity evaluation (lower level) """

        # the following lines corresponding to Algo 1 line 10 / Fig. 3(b) in the paper
        # get non-dominated architectures from archive
        F = np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # non-dominated arch bit-strings
        nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

        # initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblem(
            self.search_space, predictor, self.sec_obj,
            {'n_classes': self.n_classes, 'model_path': self.supernet_path})

        # initiate a multi-objective solver to optimize the problem
        method = get_algorithm(
            "nsga2", pop_size=40, sampling=nd_X,  # initialize with current nd archs
            crossover=get_crossover("int_two_point", prob=0.9),
            mutation=get_mutation("int_pm", eta=1.0),
            eliminate_duplicates=True)

        # kick-off the search
        res = minimize(
            problem, method, termination=('n_gen', 20), save_history=True, verbose=True)
        
        # check for duplicates
        not_duplicate = np.logical_not(
            [x in [x[0] for x in archive] for x in [self.search_space.decode(x) for x in res.pop.get("X")]])

        # the following lines corresponding to Algo 1 line 11 / Fig. 3(c)-(d) in the paper
        # form a subset selection problem to short list K from pop_size
        indices = self._subset_selection(res.pop[not_duplicate], F[front, 1], K)
        pop = res.pop[not_duplicate][indices]

        candidates = []
        for x in pop.get("X"):
            candidates.append(self.search_space.decode(x))

        # decode integer bit-string to config and also return predicted top1_err
        return candidates, predictor.predict(pop.get("X"))

    @staticmethod
    def _subset_selection(pop, nd_F, K):
        problem = SubsetProblem(pop.get("F")[:, 1], nd_F, K)
        algorithm = GA(
            pop_size=100, sampling=MySampling(), crossover=BinaryCrossover(),
            mutation=MyMutation(), eliminate_duplicates=True)

        res = minimize(
            problem, algorithm, ('n_gen', 60), verbose=False)

        return res.X

    @staticmethod
    def _calc_hv(ref_pt, F, normalized=True):
        # calculate hypervolume on the non-dominated set of F
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        ref_point = 1.01 * ref_pt
        hv = get_performance_indicator("hv", ref_point=ref_point).calc(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv


class AuxiliarySingleLevelProblem(Problem):
    """ The optimization problem for finding the next N candidate architectures """

    def __init__(self, search_space, predictor, sec_obj='flops', supernet=None):
        super().__init__(n_var=46, n_obj=2, n_constr=0, type_var=np.int)

        self.ss = search_space
        self.predictor = predictor
        self.xl = np.zeros(self.n_var)
        self.xu = 2 * np.ones(self.n_var)
        self.xu[-1] = int(len(self.ss.resolution) - 1)
        self.sec_obj = sec_obj
        self.lut = {'cpu': 'data/i7-8700K_lut.yaml'}

        # supernet engine for measuring complexity
        self.engine = OFAEvaluator(
            n_classes=supernet['n_classes'], model_path=supernet['model_path'])

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)

        top1_err = self.predictor.predict(x)[:, 0]  # predicted top1 error

        for i, (_x, err) in enumerate(zip(x, top1_err)):
            config = self.ss.decode(_x)
            subnet, _ = self.engine.sample({'ks': config['ks'], 'e': config['e'], 'd': config['d']})
            info = get_net_info(subnet, (3, config['r'], config['r']),
                                measure_latency=self.sec_obj, print_info=False, clean=True, lut=self.lut)
            f[i, 0] = err
            f[i, 1] = info[self.sec_obj]

        out["F"] = f


class SubsetProblem(Problem):
    """ select a subset to diversify the pareto front """
    def __init__(self, candidates, archive, K):
        super().__init__(n_var=len(candidates), n_obj=1,
                         n_constr=1, xl=0, xu=1, type_var=np.bool)
        self.archive = archive
        self.candidates = candidates
        self.n_max = K

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)

        for i, _x in enumerate(x):
            # append selected candidates to archive then sort
            tmp = np.sort(np.concatenate((self.archive, self.candidates[_x])))
            f[i, 0] = np.std(np.diff(tmp))
            # we penalize if the number of selected candidates is not exactly K
            g[i, 0] = (self.n_max - np.sum(_x)) ** 2

        out["F"] = f
        out["G"] = g


def main(args):
    engine = MSuNAS(vars(args))
    engine.search()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume search from a checkpoint')
    parser.add_argument('--sec_obj', type=str, default='flops',
                        help='second objective to optimize simultaneously')
    parser.add_argument('--iterations', type=int, default=30,
                        help='number of search iterations')
    parser.add_argument('--n_doe', type=int, default=100,
                        help='initial sample size for DOE')
    parser.add_argument('--n_iter', type=int, default=8,
                        help='number of architectures to high-fidelity eval (low level) in each iteration')
    parser.add_argument('--predictor', type=str, default='rbf',
                        help='which accuracy predictor model to fit (rbf/gp/cart/mlp/as)')
    parser.add_argument('--n_gpus', type=int, default=8,
                        help='total number of available gpus')
    parser.add_argument('--gpu', type=int, default=1,
                        help='number of gpus per evaluation job')
    parser.add_argument('--data', type=str, default='/mnt/datastore/ILSVRC2012',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument('--n_classes', type=int, default=1000,
                        help='number of classes of the given dataset')
    parser.add_argument('--supernet_path', type=str, default='./data/ofa_mbv3_d234_e346_k357_w1.0',
                        help='file path to supernet weights')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='number of workers for dataloader per evaluation job')
    parser.add_argument('--vld_size', type=int, default=None,
                        help='validation set size, randomly sampled from training set')
    parser.add_argument('--trn_batch_size', type=int, default=128,
                        help='train batch size for training')
    parser.add_argument('--vld_batch_size', type=int, default=200,
                        help='test batch size for inference')
    parser.add_argument('--n_epochs', type=int, default=5,
                        help='number of epochs for CNN training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluation performance on testing set')
    cfgs = parser.parse_args()
    main(cfgs)

