# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

import paddle
import paddle.nn as nn
import paddle.distributed as dist
from plsc.data import preprocess as transforms
from plsc.data import dataset as datasets
from plsc.nn import init
from visualdl import LogWriter as SummaryWriter

import plsc

import builder_moco
import vit_moco

model_names = [
    'moco_vit_small', 'moco_vit_base', 'moco_vit_conv_small',
    'moco_vit_conv_base'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument(
    '-a',
    '--arch',
    metavar='ARCH',
    default='resnet50',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) +
    ' (default: resnet50)')
parser.add_argument(
    '-j',
    '--workers',
    default=8,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 32)')
parser.add_argument(
    '--epochs',
    default=90,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=1024,
    type=int,
    metavar='N',
    help='mini-batch size (default: 1024), this is the total '
    'batch size of all GPUs on all nodes when '
    'using Data Parallel or Distributed Data Parallel')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial (base) learning rate',
    dest='lr')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--wd',
    '--weight-decay',
    default=0.,
    type=float,
    metavar='W',
    help='weight decay (default: 0.)',
    dest='weight_decay')
parser.add_argument(
    '-p',
    '--print-freq',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--world-size',
    default=-1,
    type=int,
    help='number of nodes for distributed training')
parser.add_argument(
    '--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument(
    '--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument(
    '--multiprocessing-distributed',
    action='store_true',
    help='Use multi-processing distributed training to launch '
    'N processes per node, which has N GPUs. This is the '
    'fastest way to use PyTorch for either single node or '
    'multi node data parallel training')

# additional configs:
parser.add_argument(
    '--pretrained',
    default='',
    type=str,
    help='path to moco pretrained checkpoint')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        RELATED_FLAGS_SETTING = {}
        RELATED_FLAGS_SETTING['FLAGS_cudnn_deterministic'] = 1
        paddle.fluid.set_flags(RELATED_FLAGS_SETTING)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    device = paddle.set_device("gpu")
    dist.init_parallel_env()
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    args.distributed = args.world_size > 1

    # suppress printing if not first GPU on each node
    if args.rank != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    global best_acc1

    # create model
    print("=> creating model '{}'".format(args.arch))

    model = vit_moco.__dict__[args.arch]()
    linear_keyword = 'head'

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in [
                '%s.weight' % linear_keyword, '%s.bias' % linear_keyword
        ]:
            param.stop_gradient = True

    init.normal_(getattr(model, linear_keyword).weight, mean=0.0, std=0.01)
    init.zeros_(getattr(model, linear_keyword).bias)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = paddle.load(args.pretrained)

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('base_encoder') and not k.startswith(
                        'base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.set_state_dict(state_dict)
            # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        args.batch_size = int(args.batch_size / args.world_size)
        model = paddle.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    # optimize only the linear classifier
    parameters = list(
        filter(lambda p: not p.stop_gradient, model.parameters()))
    assert len(parameters) == 2  # weight, bias

    optimizer = plsc.optimizer.Momentum(
        parameters,
        init_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = paddle.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.set_state_dict(checkpoint['state_dict'])
            optimizer.set_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normalize,
                                         ]))

    train_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, shuffle=True, batch_size=args.batch_size)

    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.workers,
        use_shared_memory=True, )

    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize,
                                       ]))

    val_sampler = paddle.io.BatchSampler(
        val_dataset, shuffle=False, batch_size=256, drop_last=False)

    val_loader = paddle.io.DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.workers,
        use_shared_memory=True, )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.batch_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.rank == 0 and epoch % 10 == 0 or epoch == args.epochs - 1:  # only the first GPU saves checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
            if epoch == args.start_epoch:
                sanity_check(model.state_dict(), args.pretrained,
                             linear_keyword)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.shape[0])
        top1.update(acc1[0].item(), images.shape[0])
        top5.update(acc5[0].item(), images.shape[0])

        # compute gradient and do SGD step
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with paddle.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.shape[0])
            top1.update(acc1[0].item(), images.shape[0])
            top5.update(acc5[0].item(), images.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
            top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pd'):
    paddle.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pd')


def sanity_check(state_dict, pretrained_weights, linear_keyword):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = paddle.load(pretrained_weights)
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = 'base_encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


@paddle.no_grad()
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.shape[1])
    batch_size = target.shape[0]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = (
        pred == target.reshape([1, -1]).expand_as(pred)).astype(paddle.float32)
    return [
        correct[:min(k, maxk)].reshape([-1]).sum(0) * 100. / batch_size
        for k in topk
    ]


if __name__ == '__main__':
    main()
