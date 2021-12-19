from __future__ import print_function, division

import math
import sys
import time
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import ImageFilter
import torch.nn as nn

from .util import AverageMeter, accuracy, is_correct_prediction
import numpy



def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()

    # set teacher as eval()
    module_list[-1].eval()

    # updates BN params
    # module_list[-1].train()

    criterion_cls = criterion_list[0]
    criterion_kd = criterion_list[1]

    model_s = module_list[0]
    model_t = module_list[-1]
    model_ta = None

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        input, target, index = data
        input = input.float().reshape(-1, 3, 32, 32)
        target = target.long().reshape(-1)

        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()

        # ===================forward=====================
        preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)

        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        loss_cls = criterion_cls(logit_s, target)

        # other kd beyond KL divergence
        y = F.one_hot(target, num_classes=opt.n_cls).float()
        f_s = feat_s[-1]
        f_t = feat_t[-1]

        loss_correlation = opt.lambda_corr * criterion_kd.forward_correlation_it(f_s, f_t)
        loss_mutual = opt.lambda_mutual * criterion_kd.forward_mutual_it(f_s, f_t)

        loss_kd = loss_mutual + loss_correlation
        loss = opt.gamma * loss_cls + opt.beta * loss_kd

        b = target.size(0)
        losses.update(loss.item(), b)
        # if not mixup:
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        top1.update(acc1[0], b)
        top5.update(acc5[0], b)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, loss_correlation, loss_mutual


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg