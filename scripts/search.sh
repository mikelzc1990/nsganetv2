#!/bin/bash

# Search Examples
## In general, set `--n_iter` to `--n_gpus`, which is the # of available gpus you have

# Maximize Top-1 Accuracy and Minimize #FLOPs on ImageNet
#python msunas.py --sec_obj flops \
#          --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 0 \
#          --data /usr/local/soft/temp-datastore/ILSVRC2012/ \
#          --predictor rbf --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
#          --save search-imagenet-flops-w1.0 --iterations 30 --vld_size 10000

# Maximize Top-1 Accuracy and Minimize #Params on ImageNet
#python msunas.py --sec_obj params \
#          --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 0 \
#          --data /usr/local/soft/temp-datastore/ILSVRC2012/ \
#          --predictor rbf --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
#          --save search-imagenet-params-w1.0 --iterations 30 --vld_size 10000

# Maximize Top-1 Accuracy and Minimize CPU Latency on ImageNet
#python msunas.py --sec_obj cpu \
#          --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 0 \
#          --data /usr/local/soft/temp-datastore/ILSVRC2012/ \
#          --predictor rbf --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
#          --save search-imagenet-cpu-w1.0 --iterations 30 --vld_size 10000

# Maximize Top-1 Accuracy and Minimize #FLOPs on CIFAR-10
#python msunas.py --sec_obj flops \
#          --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 5 \
#          --dataset cifar10 --n_classes 10 \
#          --data /usr/local/soft/temp-datastore/CIFAR/ \
#          --predictor as --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
#          --save search-cifar10-flops-w1.0 --iterations 30 --vld_size 5000

# Maximize Top-1 Accuracy and Minimize #FLOPs on CIFAR-100
#python msunas.py --sec_obj flops \
#          --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 5 \
#          --dataset cifar100 --n_classes 100 \
#          --data /usr/local/soft/temp-datastore/CIFAR/ \
#          --predictor as --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
#          --save search-cifar100-flops-w1.0 --iterations 30 --vld_size 5000

# Maximize Top-1 Accuracy and Minimize #FLOPs on CINIC10
#python msunas.py --sec_obj flops \
#          --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 5 \
#          --dataset cinic10 --n_classes 10 \
#          --data /usr/local/soft/temp-datastore/CINIC/ \
#          --predictor as --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
#          --save search-cinic10-flops-w1.0 --iterations 30 --vld_size 10000

# Maximize Top-1 Accuracy and Minimize #FLOPs on STL-10
#python msunas.py --sec_obj flops \
#          --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 5 \
#          --dataset stl10 --n_classes 10 \
#          --data /usr/local/soft/temp-datastore/STL10/ \
#          --predictor as --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
#          --save search-stl10-flops-w1.0 --iterations 30 --vld_size 500

# Maximize Top-1 Accuracy and Minimize #FLOPs on Aircraft
#python msunas.py --sec_obj flops \
#          --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 5 \
#          --dataset aircraft --n_classes 100 \
#          --data /usr/local/soft/temp-datastore/Aircraft/ \
#          --predictor as --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
#          --save search-aircraft-flops-w1.0 --iterations 30 --vld_size 500

# Random Search on ImageNet (set `--n_doe` to the total # of archs you want to sample)
#python msunas.py --n_doe 350 --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 0 \
#          --data /usr/local/soft/temp-datastore/ILSVRC2012/ \
#          --predictor rbf --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
#          --save random-imagenet-w1.0 --iterations 0 --vld_size 10000

# Random Search on CIFAR-10 (set `--n_doe` to the total # of archs you want to sample)
#python msunas.py --n_doe 350 --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 0 \
#          --dataset cifar10 --n_classes 10 \
#          --data /usr/local/soft/temp-datastore/CIFAR/ \
#          --predictor as --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
#          --save random-cifar10-w1.0 --iterations 0 --vld_size 5000

