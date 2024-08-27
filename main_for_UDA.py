# -*- coding: utf-8 -*-
"""
@author: 

"""
import torch
import numpy as np
import random
import os
import argparse

from loader.sampler import LabelSampler
from loader.data_loader import load_data_for_UDA
from model.model import JOINT
from utils.optimizer import get_optimizer
from utils.train_UDA import finetune_for_UDA, train_for_UDA
from utils.eval import predict
from utils import globalvar as gl
# import dataloader as dir_dataloader
parser = argparse.ArgumentParser(description='UDA Classification')
parser.add_argument('--dataset', type=str, default='officehome',
                    choices=['multi', 'office31', 'officehome','imageclef','visda2017', 'pacs'],
                    help='the name of dataset')
parser.add_argument('--source', type=str, default='Art',
                    help='source domain')
parser.add_argument('--target', type=str, default='Product',
                    help='target domain')
parser.add_argument('--net', type=str, default='resnet',
                    choices=['alexnet', 'vgg16', 'resnet34', 'resnet', 'resnet18'],
                    help='which network to use')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--phase', type=str, default='train',
                    choices=['pretrain', 'train', 'test', 'test_upper_bound'],
                    help='the phase of training model')
parser.add_argument('--load_model', type=str, default='',
                    help='the pretrain model to train')
parser.add_argument('--root_dir', type=str, default='./data',
                    help='root dir of the dataset')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr_mult', type=float, nargs=4, default=[0.1, 0.1, 1, 1],
                    help='lr_mult (default: [0.1, 0.1, 1, 1])')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to pretrain (default: 30)')
parser.add_argument('--steps', type=int, default=50000,
                    help='maximum number of iterations to train (default: 50000)')
parser.add_argument('--lam_step', type=int, default=20000,
                    help='factor of lamda (default: 20000)')
parser.add_argument('--log_interval', type=int, default=100,
                    help='how many batches to wait before logging training status(default: 100)')
parser.add_argument('--save_interval', type=int, default=500,
                    help='how many batches to wait before saving a model(default: 500)')
parser.add_argument('--update_interval', type=int, default=1000,
                    help='how many batches to wait before updating pseudo labels(default: 1000)')
parser.add_argument('--start_update_step', type=int, default=2000,
                    help='how many batches to wait before the first time to update the pseudo labels(default: 2000)')
parser.add_argument('--save_check', type=bool, default=True,
                    help='save checkpoint or not(default: True)')
parser.add_argument('--patience', type=int, default=25,
                    help='early stopping to wait for improvment before terminating. (default: 25 (12500 iterations))')
parser.add_argument('--early', type=bool, default=True,
                    help='early stopping or not(default: True)')
parser.add_argument("--pre_type", default=1, type=int, help="pre-processing")
parser.add_argument('--momentum', type=float, default=0.9,
                    help='MOMENTUM of SGD (default: 0.9)')
parser.add_argument('--decay', type=int, default=0.0005,
                    help='DECAY of SGD (default: 0.0005)')
parser.add_argument('--alpha_div', type=float, default=0.5,
                    help='control the trade-off of two parts, note that 0<=alpha<=1')     
parser.add_argument('--beta_div', type=float, default=0.5,
                    help='control the radio of the distribution, note that 0<=beta<=1')  
parser.add_argument('--lambda_div', type=float, default=0.01,
                    help='the penalty factor in the estimation of divergence')    
parser.add_argument("--batch_size", default=60, type=int, help="batch size (discarded)") 


# ===========================================================================================================================================================================
args = parser.parse_args()
# args.root_dir = '/data' # YOUR PATH



DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
gl._init()
gl.set_value('DEVICE', DEVICE)
domain_name = {
    'officehome': ['Art','Clipart','Product','Real'],
    'imageclef':['c','i','p'],
    'office31':['amazon','dslr','webcam'],
    'multi': ['clipart', 'painting', 'real', 'sketch'],
    'visda2017':['synthetic','real'],
    'pacs':['art_painting', 'cartoon', 'photo', 'sketch']
}[args.dataset]

class_num = {
    'officehome': 65,
    'imageclef': 12,
    'office31': 31,
    'multi': 126,
    'visda2017': 12,
    'pacs': 7
}[args.dataset]

bottleneck_dim = {
    'alexnet': 1024, 
    'vgg16': 1024,
    'resnet34': 256*4,
    'resnet': 2048,
    "resnet18": 1024
}[args.net]

if args.dataset in ['visda2017']:
    bottleneck_dim = {
        'alexnet': 1024, 
        'vgg16': 1024,
        'resnet34': 256*4,
        'resnet': 512,
        "resnet18": 1024
    }[args.net]
# batch_size of src_l and tar_ul are both batch_size
if args.batch_size is None:
    batch_size = 48
else:
    batch_size = {
        'alexnet': 128, 
        'vgg16': 48,
        'resnet34': 96,
        'resnet': 60, #96,
        "resnet18": 96
    }[args.net]

args.min_scale=0.8
args.max_scale=1.0
args.flip=0.5
args.jitter=0.4

select_class_num = batch_size // 2 
while select_class_num > class_num:
    select_class_num //= 2

seed = 10
torch.manual_seed(seed)
random.seed(seed)


record_dir = 'record_UDA/{}/{}'.format(str.upper(args.dataset), str.upper(args.net))
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
check_path = 'save_model_UDA/{}/{}'.format(str.upper(args.dataset), str.upper(args.net))
# check_path = 'save_parallel_model_UDA/{}/{}'.format(str.upper(args.dataset), str.upper(args.net))
if not os.path.exists(check_path):
    os.makedirs(check_path)
if args.phase == 'pretrain':
    record_file = os.path.join(record_dir, 'pretrain_{}_{}.txt'.format(args.net, args.source))
else:
    record_file = os.path.join(record_dir, 'JOINT_{}_{}_to_{}.txt'.format(args.net, args.source, args.target))

gl.set_value('check_path', check_path)
gl.set_value('record_file', record_file)

if __name__ == '__main__':
    dataloaders = {}
    model = JOINT(args.net, class_num, bottleneck_dim, args.alpha_div, args.beta_div, args.lambda_div).to(DEVICE)
    # model = torch.nn.DataParallel(model)

    if args.phase == 'pretrain':
        dataloaders['src_pretrain'] = load_data_for_UDA(
            args.root_dir, args.dataset, args.source, args.source, args.phase, batch_size, args.net, args=args)
        dataloaders['src_test'], dataloaders['tar_test'] = load_data_for_UDA(
            args.root_dir, args.dataset, args.source, args.source, 'test', batch_size, args.net, args=args)

        print(len(dataloaders['src_pretrain'].dataset))
        print(len(dataloaders['src_test'].dataset))
        print(len(dataloaders['tar_test'].dataset))
        optimizer = get_optimizer(model, args.lr, args.lr_mult)

        finetune_for_UDA(args, model, optimizer, dataloaders)

    elif args.phase == 'train':
        model_path = '{}/best(PT)_{}_{}.pth'.format(check_path, args.net, args.source)
        print('model_path:{}'.format(model_path))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        if args.start_update_step <= args.update_interval:
            first_batch_num = args.update_interval
        else:
            a = args.start_update_step % args.update_interval
            first_batch_num = args.start_update_step if a == 0  else (args.start_update_step // args.update_interval + 1) * args.update_interval
        label_sampler = LabelSampler(first_batch_num, class_num, select_class_num)
        dataloaders['src_train_l'], dataloaders['tar_train_ul'] = load_data_for_UDA(
            args.root_dir, args.dataset, args.source, args.target, args.phase, batch_size, args.net, label_sampler, pre_type=args.pre_type, args=args)
        dataloaders['src_test'], dataloaders['tar_test'] = load_data_for_UDA(
            args.root_dir, args.dataset, args.source, args.target, 'test', batch_size, args.net, pre_type=args.pre_type, args=args)
        pseudo_labels = predict(model, dataloaders['tar_test'])
        dataloaders['tar_train_ul'].dataset.update_pseudo_labels(pseudo_labels)

        print(len(dataloaders['src_train_l'].dataset), len(dataloaders['tar_train_ul'].dataset))
        optimizer = get_optimizer(model, args.lr, args.lr_mult, args.momentum, args.decay)
        train_for_UDA(args, model, optimizer, dataloaders)
    