import os
import copy
import json
import yaml
import numpy as np
from collections import OrderedDict
from torchprofile import profile_macs

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover

DEFAULT_CFG = {
    'gpus': '0', 'config': None, 'init': None, 'trn_batch_size': 128, 'vld_batch_size': 250, 'num_workers': 4,
    'n_epochs': 0, 'save': None, 'resolution': 224, 'valid_size': 10000, 'test': True, 'latency': None,
    'verbose': False, 'classifier_only': False, 'reset_running_statistics': True,
}


def get_correlation(prediction, target):
    import scipy.stats as stats

    rmse = np.sqrt(((prediction - target) ** 2).mean())
    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, rho, tau


def bash_command_template(**kwargs):
    gpus = kwargs.pop('gpus', DEFAULT_CFG['gpus'])
    cfg = OrderedDict()
    cfg['subnet'] = kwargs['subnet']
    cfg['data'] = kwargs['data']
    cfg['dataset'] = kwargs['dataset']
    cfg['n_classes'] = kwargs['n_classes']
    cfg['supernet_path'] = kwargs['supernet_path']
    cfg['config'] = kwargs.pop('config', DEFAULT_CFG['config'])
    cfg['init'] = kwargs.pop('init', DEFAULT_CFG['init'])
    cfg['save'] = kwargs.pop('save', DEFAULT_CFG['save'])
    cfg['trn_batch_size'] = kwargs.pop('trn_batch_size', DEFAULT_CFG['trn_batch_size'])
    cfg['vld_batch_size'] = kwargs.pop('vld_batch_size', DEFAULT_CFG['vld_batch_size'])
    cfg['num_workers'] = kwargs.pop('num_workers', DEFAULT_CFG['num_workers'])
    cfg['n_epochs'] = kwargs.pop('n_epochs', DEFAULT_CFG['n_epochs'])
    cfg['resolution'] = kwargs.pop('resolution', DEFAULT_CFG['resolution'])
    cfg['valid_size'] = kwargs.pop('valid_size', DEFAULT_CFG['valid_size'])
    cfg['test'] = kwargs.pop('test', DEFAULT_CFG['test'])
    cfg['latency'] = kwargs.pop('latency', DEFAULT_CFG['latency'])
    cfg['verbose'] = kwargs.pop('verbose', DEFAULT_CFG['verbose'])
    cfg['classifier_only'] = kwargs.pop('classifier_only', DEFAULT_CFG['classifier_only'])
    cfg['reset_running_statistics'] = kwargs.pop(
        'reset_running_statistics', DEFAULT_CFG['reset_running_statistics'])

    execution_line = "CUDA_VISIBLE_DEVICES={} python evaluator.py".format(gpus)
    for k, v in cfg.items():
        if v is not None:
            if isinstance(v, bool):
                if v:
                    execution_line += " --{}".format(k)
            else:
                execution_line += " --{} {}".format(k, v)
    execution_line += ' &'
    return execution_line


def prepare_eval_folder(path, configs, gpu=2, n_gpus=8, **kwargs):
    """ create a folder for parallel evaluation of a population of architectures """
    os.makedirs(path, exist_ok=True)
    gpu_template = ','.join(['{}'] * gpu)
    gpus = [gpu_template.format(i, i + 1) for i in range(0, n_gpus, gpu)]
    bash_file = ['#!/bin/bash']
    for i in range(0, len(configs), n_gpus//gpu):
        for j in range(n_gpus//gpu):
            if i + j < len(configs):
                job = os.path.join(path, "net_{}.subnet".format(i + j))
                with open(job, 'w') as handle:
                    json.dump(configs[i + j], handle)
                bash_file.append(bash_command_template(
                    gpus=gpus[j], subnet=job, save=os.path.join(
                        path, "net_{}.stats".format(i + j)), **kwargs))
        bash_file.append('wait')

    with open(os.path.join(path, 'run_bash.sh'), 'w') as handle:
        for line in bash_file:
            handle.write(line + os.linesep)


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=np.bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X


class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            try:
                X[i, np.random.choice(is_false)] = True
                X[i, np.random.choice(is_true)] = False
            except ValueError:
                pass

        return X


class LatencyEstimator(object):
    """
    Modified from https://github.com/mit-han-lab/proxylessnas/blob/
    f273683a77c4df082dd11cc963b07fc3613079a0/search/utils/latency_estimator.py#L29
    """
    def __init__(self, fname):
        # fname = download_url(url, overwrite=True)

        with open(fname, 'r') as fp:
            self.lut = yaml.load(fp, yaml.SafeLoader)

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def predict(self, ltype: str, _input, output, expand=None,
                kernel=None, stride=None, idskip=None, se=None):
        """
        :param ltype:
            Layer type must be one of the followings
                1. `first_conv`: The initial stem 3x3 conv with stride 2
                2. `final_expand_layer`: (Only for MobileNet-V3)
                    The upsample 1x1 conv that increases num_filters by 6 times + GAP.
                3. 'feature_mix_layer':
                    The upsample 1x1 conv that increase num_filters to num_features + torch.squeeze
                3. `classifier`: fully connected linear layer (num_features to num_classes)
                4. `MBConv`: MobileInvertedResidual
        :param _input: input shape (h, w, #channels)
        :param output: output shape (h, w, #channels)
        :param expand: expansion ratio
        :param kernel: kernel size
        :param stride:
        :param idskip: indicate whether has the residual connection
        :param se: indicate whether has squeeze-and-excitation
        """
        infos = [ltype, 'input:%s' % self.repr_shape(_input),
                 'output:%s' % self.repr_shape(output), ]
        if ltype in ('MBConv',):
            assert None not in (expand, kernel, stride, idskip, se)
            infos += ['expand:%d' % expand, 'kernel:%d' % kernel,
                      'stride:%d' % stride, 'idskip:%d' % idskip, 'se:%d' % se]
        key = '-'.join(infos)
        return self.lut[key]['mean']


def look_up_latency(net, lut, resolution=224):
    def _half(x, times=1):
        for _ in range(times):
            x = np.ceil(x / 2)
        return int(x)

    predicted_latency = 0

    # first_conv
    predicted_latency += lut.predict(
        'first_conv', [resolution, resolution, 3],
        [resolution // 2, resolution // 2, net.first_conv.out_channels])

    # final_expand_layer (only for MobileNet V3 models)
    input_resolution = _half(resolution, times=5)
    predicted_latency += lut.predict(
        'final_expand_layer',
        [input_resolution, input_resolution, net.final_expand_layer.in_channels],
        [input_resolution, input_resolution, net.final_expand_layer.out_channels]
    )

    # feature_mix_layer
    predicted_latency += lut.predict(
        'feature_mix_layer',
        [1, 1, net.feature_mix_layer.in_channels],
        [1, 1, net.feature_mix_layer.out_channels]
    )

    # classifier
    predicted_latency += lut.predict(
        'classifier',
        [net.classifier.in_features],
        [net.classifier.out_features]
    )

    # blocks
    fsize = _half(resolution)
    for block in net.blocks:
        idskip = 0 if block.config['shortcut'] is None else 1
        se = 1 if block.config['mobile_inverted_conv']['use_se'] else 0
        stride = block.config['mobile_inverted_conv']['stride']
        out_fz = _half(fsize) if stride > 1 else fsize
        block_latency = lut.predict(
            'MBConv',
            [fsize, fsize, block.config['mobile_inverted_conv']['in_channels']],
            [out_fz, out_fz, block.config['mobile_inverted_conv']['out_channels']],
            expand=block.config['mobile_inverted_conv']['expand_ratio'],
            kernel=block.config['mobile_inverted_conv']['kernel_size'],
            stride=stride, idskip=idskip, se=se
        )
        predicted_latency += block_latency
        fsize = out_fz

    return predicted_latency


def get_net_info(net, input_shape=(3, 224, 224), measure_latency=None, print_info=True, clean=False, lut=None):
    """
    Modified from https://github.com/mit-han-lab/once-for-all/blob/
    35ddcb9ca30905829480770a6a282d49685aa282/ofa/imagenet_codebase/utils/pytorch_utils.py#L139
    """
    from ofa.imagenet_codebase.utils.pytorch_utils import count_parameters, measure_net_latency

    # artificial input data
    inputs = torch.randn(1, 3, input_shape[-2], input_shape[-1])

    # move network to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        net = net.to(device)
        cudnn.benchmark = True
        inputs = inputs.to(device)

    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module

    # parameters
    net_info['params'] = count_parameters(net)

    # flops
    net_info['flops'] = int(profile_macs(copy.deepcopy(net), inputs))

    # latencies
    latency_types = [] if measure_latency is None else measure_latency.split('#')

    # print(latency_types)
    for l_type in latency_types:
        if lut is not None and l_type in lut:
            latency_estimator = LatencyEstimator(lut[l_type])
            latency = look_up_latency(net, latency_estimator, input_shape[2])
            measured_latency = None
        else:
            latency, measured_latency = measure_net_latency(
                net, l_type, fast=False, input_shape=input_shape, clean=clean)
        net_info['%s latency' % l_type] = {
            'val': latency,
            'hist': measured_latency
        }

    if print_info:
        # print(net)
        print('Total training params: %.2fM' % (net_info['params'] / 1e6))
        print('Total FLOPs: %.2fM' % (net_info['flops'] / 1e6))
        for l_type in latency_types:
            print('Estimated %s latency: %.3fms' % (l_type, net_info['%s latency' % l_type]['val']))

    return net_info
