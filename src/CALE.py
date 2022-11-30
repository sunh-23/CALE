import sys
import time
import copy
import shutil
import random
import warnings
import argparse
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from dalib.modules.domain_discriminator import DomainDiscriminator
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger

sys.path.append('.')
import utils.misc as misc
from modules.classifier import Classifier
from modules import divergence as D
from modules.regulization import CooperativeAdversarialLearning as CALE
from modules.cdan import ConditionalDomainAdversarialLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    src_train_transform, tgt_train_transform, val_transform = misc.get_transforms(args)
    print("src_train_transform: ", src_train_transform)
    print("tgt_train_transform: ", tgt_train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        misc.get_dataset(args.data, args.root, args.source, args.target, src_train_transform, val_transform, tgt_train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = misc.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = Classifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier_feature_dim = classifier.features_dim

    if args.randomized:
        domain_discri = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)

    all_parameters = classifier.get_parameters() + domain_discri.get_parameters()

    # define optimizer and lr scheduler
    optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)

    cale = CALE(classifier.bottleneck, classifier.head, domain_adv, D.CrossEntropy(), args)

    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        acc1 = misc.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, cale, domain_adv, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = misc.validate(val_loader, classifier, args, device, epoch)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = misc.validate(test_loader, classifier, args, device, args.epochs)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: Classifier, cale: CALE, domain_adv: ConditionalDomainAdversarialLoss,
          optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):

    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    reg_losses = AverageMeter("Reg Loss", ":3.2f")
    fixmatch_losses = AverageMeter("FixMatch Loss", ":3.2f")
    src_accs = AverageMeter('Src Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, trans_losses, reg_losses, fixmatch_losses, domain_accs, src_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        (x_s_weak, x_s_strong), labels_s = next(train_source_iter)
        (x_t_weak, x_t_strong), labels_t = next(train_target_iter)

        x_s_weak,   x_t_weak   = x_s_weak.to(device),   x_t_weak.to(device)
        x_s_strong, x_t_strong = x_s_strong.to(device), x_t_strong.to(device)
        labels_s,   labels_t   = labels_s.to(device),   labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        # forward
        x = torch.cat((x_s_weak, x_s_strong, x_t_weak, x_t_strong), dim=0) # need to check
        labels_s = torch.cat((labels_s, labels_s), dim=0)
        labels_t = torch.cat((labels_t, labels_t), dim=0)

        batch_size = labels_s.size(0)

        y, f1, f2 = model(x)
        y_s, y_t = y.chunk(2, dim=0)

        # loss
        # disc. loss
        y_t_weak, y_t_strong = y_t.chunk(2, dim=0)
        preds = torch.softmax(y_t_weak.detach(), dim=-1)
        max_probs, pseudo_label = torch.max(preds, dim=-1)
        Lu = (
            F.cross_entropy(y_t_strong, pseudo_label, reduction="none") * max_probs.ge(args.fixmatch_threshold).float()
        ).sum() / batch_size
        cls_loss = F.cross_entropy(y_s, labels_s) + Lu
        # tran. loss
        d, transfer_loss = domain_adv(y, f2)
        # CALE regularization loss
        reg_loss = cale.regulization_loss(f1, y, d, y.detach(), d.detach())

        # total loss
        loss = cls_loss + transfer_loss + reg_loss * args.trade_off

        #logs
        losses.update(loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)
        fixmatch_losses.update(Lu.item(), batch_size)
        trans_losses.update(transfer_loss.item(), batch_size)
        reg_losses.update(reg_loss.item(), batch_size)

        src_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]
        domain_acc = domain_adv.domain_discriminator_accuracy

        src_accs.update(src_acc, batch_size // 2)
        tgt_accs.update(tgt_acc, batch_size // 2)
        domain_accs.update(domain_acc, batch_size // 2)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if epoch < 30:
            lr_scheduler.step()
            domain_adv.grl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cooperative and Adversarial LEarning for Discriminability and Transferability in Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=misc.get_dataset_names(),
                        help='dataset: ' + ' | '.join(misc.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=misc.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(misc.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('-r', '--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=True, action='store_false', help='use entropy conditioning')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for CALE regulization loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 64)')
    parser.add_argument('--sgd-momentum', default=0.9, type=float, metavar='M',
                        dest='momentum',
                        help='momentum, valid if optimizer is sgd')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    parser.add_argument('--fixmatch_threshold', default=0.95, type=float)
    parser.add_argument('--eps', default=1., type=float, help='')

    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cdan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    args = parser.parse_args()
    main(args)

