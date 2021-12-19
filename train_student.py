"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time
from shutil import copyfile

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed, ConvEnc
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from helper.util import adjust_learning_rate
from distiller_zoo import ITLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')

    # CIFAR100
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')

    # ImageNet
    # parser.add_argument('--batch_size', type=int, default=256, help='batch_size')

    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet34', 'resnet18', # ImageNet
                                 'resnet20_1w1a', # binary
                                 'resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])

    parser.add_argument('--path_s', type=str, default=None, help='student model snapshot')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    parser.add_argument('--kd_T', type=float, default=4.0, help='temperature for KD distillation')

    # IT distillation
    parser.add_argument('--lambda_corr', type=float, default=2.0, help='correlation loss weight')
    parser.add_argument('--lambda_mutual', type=float, default=0.05, help='mutual information loss weight')
    parser.add_argument('--alpha_it', type=float, default=1.50, help='Renyis alpha')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_r:{}_b:{}_{}_l_corr:{}_l_mutual:{}_alpha:{}'.format(opt.model_s, opt.model_t, opt.dataset,
                                                                                          opt.gamma, opt.beta, opt.trial,
                                                                                          opt.lambda_corr, opt.lambda_mutual, opt.alpha_it)
    # : is not allowed in Windows paths
    opt.model_name = opt.model_name.replace(":", "-")

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.epochs = opt.epochs
    opt.lr_decay_epochs = [int(epoch) for epoch in opt.lr_decay_epochs]
    opt.batch_size = int(opt.batch_size)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def load_teacher(model_path, n_cls, use_attention=False):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'], strict=False)
    print('==> done')
    return model

def main():
    best_acc = 0

    opt = parse_option()

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                           num_workers=opt.num_workers)
        n_cls = 100

    else:
        raise NotImplementedError(opt.dataset)

    opt.n_cls = n_cls
    model_t = model_dict[opt.model_t](num_classes=n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    # do not need to load teacher weights when just doing repr transfer
    if opt.path_t is not None:
        # addressing some compatibility issues [When loading ssd weights]
        # weights = torch.load(opt.path_t)['state_dict']
        weights = torch.load(opt.path_t)
        
        if 'model' in weights:
            # CIFAR100, ckpr_epoch_240.pth have the weights inside 'model'
            weights = weights['model']

        # loading HSAKD weights
        if 'net' in weights:
            weights = weights['net']
            weights_out = {}
            for key in weights:
                if 'auxiliary' not in key:
                    key_n = key.replace('backbone.','')
                    weights_out[key_n] = weights[key]
            weights = weights_out
                
        model_t.load_state_dict(weights, strict=False)

    # similarly for the student model
    if opt.path_s is not None:
        weights = torch.load(opt.path_s)
        if 'model' in weights:
            weights = weights['model']
        model_s.load_state_dict(weights, strict=False)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    # additional trainable parameters
    trainable_list_params = []
    criterion_cls = nn.CrossEntropyLoss()

    opt.s_dim = feat_s[-1].shape[1]
    opt.t_dim = feat_t[-1].shape[1]
    opt.n_data = n_data

    criterion_kd = ITLoss(opt)

    module_list.append(criterion_kd)
    trainable_list.append(criterion_kd)

    # Lcorr+Lmi share embed
    module_list.append(criterion_kd.embed)
    trainable_list.append(criterion_kd.embed)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(list(trainable_list.parameters()) + trainable_list_params,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    with open(os.path.join(opt.save_folder, 'log.txt'), "w") as log_file:
        log_file.write('test_acc, test_loss, test_acc_top5\n')

    # copy python files for easy reference
    copyfile("train_student.py", os.path.join(opt.save_folder, 'train_student.py'))
    copyfile("distiller_zoo/IT.py", os.path.join(opt.save_folder, 'IT.py'))
    copyfile("helper/loops.py", os.path.join(opt.save_folder, 'loops.py')) 

    start_epoch = 1
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        # for i in range(4):
        train_acc, train_loss, correlation_loss, mutual_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('train_correlation_loss', correlation_loss, epoch)
        logger.log_value('train_mutual_loss', mutual_loss, epoch)
        
        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)

        with open(os.path.join(opt.save_folder, 'log.txt'), "a") as log_file:
            log_file.write("{}, {}, {}\n".format(test_acc, test_loss, test_acc_top5))

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            } 
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)

    with open(os.path.join(opt.save_folder, 'log.txt'), "a") as log_file:
        log_file.write("------")
        log_file.write('best accuracy: {}\n'.format(best_acc))

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
