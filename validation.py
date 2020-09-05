import time
import json
import torch
import logging
import argparse
from collections import OrderedDict

from timm.utils import accuracy, AverageMeter, setup_default_logging

from codebase.run_manager import get_run_config
from codebase.networks.nsganetv2 import NSGANetV2


def validate(model, loader, criterion, log_freq=50):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            target = target.cuda()
            input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_freq == 0:
                logging.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        i, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses, top1=top1, top5=top5))

    results = OrderedDict(
        top1=round(top1.avg, 4), top1_err=round(100 - top1.avg, 4),
        top5=round(top5.avg, 4), top5_err=round(100 - top5.avg, 4))

    logging.info(' * Acc@1 {:.1f} ({:.3f}) Acc@5 {:.1f} ({:.3f})'.format(
        results['top1'], results['top1_err'], results['top5'], results['top5_err']))


def main(args):
    setup_default_logging()

    logging.info('Running validation on {}'.format(args.dataset))

    net_config = json.load(open(args.model))
    if 'img_size' in net_config:
        img_size = net_config['img_size']
    else:
        img_size = args.img_size

    run_config = get_run_config(
        dataset=args.dataset, data_path=args.data, image_size=img_size, n_epochs=0,
        train_batch_size=args.batch_size, test_batch_size=args.batch_size,
        n_worker=args.workers, valid_size=None)

    model = NSGANetV2.build_from_config(net_config)

    param_count = sum([m.numel() for m in model.parameters()])
    logging.info('Model created, param count: %d' % param_count)

    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    validate(model, run_config.test_loader, criterion)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data related settings
    parser.add_argument('--data', type=str, default='/mnt/datastore/ILSVRC2012',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument('-j', '--workers', type=int, default=6,
                        help='number of workers for data loading')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='test batch size for inference')
    parser.add_argument('--img-size', type=int, default=224,
                        help='input resolution (192 -> 256)')
    # model related settings
    parser.add_argument('--model', '-m', metavar='MODEL', default='', type=str,
                        help='model configuration file')
    parser.add_argument('--pretrained', type=str, default='',
                        help='path to pretrained weights')
    cfgs = parser.parse_args()

    main(cfgs)
