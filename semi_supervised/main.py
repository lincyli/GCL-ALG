import re
import sys
import argparse
import time
import random
import numpy as np
import os
import shutil
import glob
import logging
from itertools import product

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from semi_supervised.train_eval import joint_cl_exp,graph_cl_exp,benchmark_exp
from semi_supervised.res_gcn import ResGCN

sys.path.append(os.path.abspath(os.path.join('..')))
from utils import logger
from datasets import get_dataset

str2bool = lambda x: x.lower() == "true"

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1304)
parser.add_argument('--exp', type=str, default="joint_cl_exp")
parser.add_argument('--data_root', type=str, default="../data")
parser.add_argument('--dataset', type=str, default="COLLAB")

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--save', type=str, default="EXP")

parser.add_argument('--n_fold', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=1.0)
parser.add_argument('--lr_decay_step_size', type=int, default=20)
parser.add_argument('--epoch_select', type=str, default='test_max')
parser.add_argument('--n_layers_feat', type=int, default=1)
parser.add_argument('--n_layers_conv', type=int, default=3)
parser.add_argument('--n_layers_fc', type=int, default=2)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--global_pool', type=str, default="sum")
parser.add_argument('--skip_connection', type=str2bool, default=False)
parser.add_argument('--res_branch', type=str, default="BNConvReLU")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--edge_norm', type=str2bool, default=True)
parser.add_argument('--with_eval_mode', type=str2bool, default=True)
parser.add_argument('--add_mask', type=str2bool, default=False)

parser.add_argument('--aug_ratio', type=float, default=0.2)
parser.add_argument('--semi_split', type=int, default=10, help='percent of semi training data')
parser.add_argument('--local_rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
args = parser.parse_args()


def create_n_filter_triples(datasets, feat_strs, nets, gfn_add_ak3=False,
                            gfn_reall=True, reddit_odeg10=False,
                            dd_odeg10_ak1=False):
    # <class 'list'>: [('MUTAG', 'deg+odeg100', 'ResGCN')]
    triples = [(d, f, n) for d, f, n in product(datasets, feat_strs, nets)]

    triples_filtered = []
    for dataset, feat_str, net in triples:
        # Add ak3 for GFN.
        if gfn_add_ak3 and 'GFN' in net:
            feat_str += '+ak3'
        # Remove edges for GFN.
        if gfn_reall and 'GFN' in net:
            feat_str += '+reall'
        # Replace degree feats for REDDIT datasets (less redundancy, faster).
        if reddit_odeg10 and dataset in [
            'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
        # Replace degree and akx feats for dd (less redundancy, faster).
        if dd_odeg10_ak1 and dataset in ['DD']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
            feat_str = feat_str.replace('ak3', 'ak1')
        triples_filtered.append((dataset, feat_str, net))
    return triples_filtered


def get_model_with_default_configs(model_name,
                                   num_feat_layers=args.n_layers_feat,
                                   num_conv_layers=args.n_layers_conv,
                                   num_fc_layers=args.n_layers_fc,
                                   residual=args.skip_connection,
                                   hidden=args.hidden):
    # More default settings.
    res_branch = args.res_branch
    global_pool = args.global_pool
    dropout = args.dropout
    edge_norm = args.edge_norm

    # modify default architecture when needed
    if model_name.find('_') > 0:
        num_conv_layers_ = re.findall('_conv(\d+)', model_name)
        if len(num_conv_layers_) == 1:
            num_conv_layers = int(num_conv_layers_[0])
            print('[INFO] num_conv_layers set to {} as in {}'.format(
                num_conv_layers, model_name))
        num_fc_layers_ = re.findall('_fc(\d+)', model_name)
        if len(num_fc_layers_) == 1:
            num_fc_layers = int(num_fc_layers_[0])
            print('[INFO] num_fc_layers set to {} as in {}'.format(
                num_fc_layers, model_name))
        residual_ = re.findall('_res(\d+)', model_name)
        if len(residual_) == 1:
            residual = bool(int(residual_[0]))
            print('[INFO] residual set to {} as in {}'.format(
                residual, model_name))
        gating = re.findall('_gating', model_name)
        if len(gating) == 1:
            global_pool += "_gating"
            print('[INFO] add gating to global_pool {} as in {}'.format(
                global_pool, model_name))
        dropout_ = re.findall('_drop([\.\d]+)', model_name)
        if len(dropout_) == 1:
            dropout = float(dropout_[0])
            print('[INFO] dropout set to {} as in {}'.format(
                dropout, model_name))
        hidden_ = re.findall('_dim(\d+)', model_name)
        if len(hidden_) == 1:
            hidden = int(hidden_[0])
            print('[INFO] hidden set to {} as in {}'.format(
                hidden, model_name))

    if model_name.startswith('ResGFN'):
        collapse = True if 'flat' in model_name else False

        def foo(dataset):
            return ResGCN(dataset, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=True, collapse=collapse,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    elif model_name.startswith('ResGCN'):
        def foo(dataset):
            return ResGCN(dataset, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=False, collapse=False,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    else:
        raise ValueError("Unknown model {}".format(model_name))
    return foo

def run_benchmark_exp(args, device, logger):
    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    nets = ['ResGCN']
    dataset_feat_net_triples = create_n_filter_triples(datasets, feat_strs, nets, gfn_add_ak3=True, reddit_odeg10=True,
                                                       dd_odeg10_ak1=True)

    results = []
    exp_nums = len(dataset_feat_net_triples)

    logger.info("-----\nTotal %d experiments in this run:" % exp_nums)
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        logger.info('{}/{} - {} - {} - {}'.format(
            exp_id + 1, exp_nums, dataset_name, feat_str, net))
    sys.stdout.flush()

    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        logger.info('-----\n{}/{} - {} - {} - {}'.format(
            exp_id + 1, exp_nums, dataset_name, feat_str, net))
        sys.stdout.flush()
        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)

        model_func = get_model_with_default_configs(net)
        train_acc, acc, std, duration = benchmark_exp(
            device,
            logger,
            dataset,
            model_func,
            folds=args.n_fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            epoch_select=args.epoch_select,
            with_eval_mode=args.with_eval_mode,
            semi_split=args.semi_split)

        summary1 = 'data={}, model={}, feat={}, eval={}'.format(
            dataset_name, net, feat_str, args.epoch_select)
        summary2 = 'train_acc={:.2f}, test_acc={:.2f} ± {:.2f}, sec={}'.format(
            train_acc * 100, acc * 100, std * 100, round(duration, 2))
        results += ['{}: {}, {}'.format('fin-result', summary1, summary2)]
        logger.info('{}: {}, {}'.format('mid-result', summary1, summary2))
    logger.info('-----\n{}'.format('\n'.join(results)))

def run_graph_cl_exp(args, device, logger):
    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    nets = ['ResGCN']
    dataset_feat_net_triples = create_n_filter_triples(datasets, feat_strs, nets,
                                                       gfn_add_ak3=True,
                                                       reddit_odeg10=True,
                                                       dd_odeg10_ak1=True)

    results = []
    exp_nums = len(dataset_feat_net_triples)

    logger.info("-----\nTotal %d experiments in this run:" % exp_nums)
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        logger.info('{}/{} - {} - {} - {}'.format(
            exp_id + 1, exp_nums, dataset_name, feat_str, net))
    sys.stdout.flush()

    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        logger.info('-----\n{}/{} - {} - {} - {}'.format(
            exp_id + 1, exp_nums, dataset_name, feat_str, net))
        sys.stdout.flush()
        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
        # continue

        model_func = get_model_with_default_configs(net)
        train_acc, acc, std, duration = graph_cl_exp(
            device,
            logger,
            dataset,
            model_func,
            folds=args.n_fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            epoch_select=args.epoch_select,
            with_eval_mode=args.with_eval_mode,
            semi_split=args.semi_split,
            aug_ratio=args.aug_ratio)

        summary1 = 'data={}, model={}, feat={}, eval={}'.format(
            dataset_name, net, feat_str, args.epoch_select)
        summary2 = 'train_acc={:.2f}, test_acc={:.2f} ± {:.2f}, sec={}'.format(
            train_acc * 100, acc * 100, std * 100, round(duration, 2))
        results += ['{}: {}, {}'.format('fin-result', summary1, summary2)]
        logger.info('{}: {}, {}'.format('mid-result', summary1, summary2))
        # sys.stdout.flush()
    logger.info('-----\n{}'.format('\n'.join(results)))

def run_joint_cl_exp(args, device, logger):
    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    nets = ['ResGCN']
    # <class 'list'>: [('MUTAG', 'deg+odeg100', 'ResGCN')]
    dataset_feat_net_triples = create_n_filter_triples(datasets, feat_strs, nets,
                                                       gfn_add_ak3=True,
                                                       reddit_odeg10=True,
                                                       dd_odeg10_ak1=True)
    dist.init_process_group(backend='nccl')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    # dataset = TUDataset(args.dataset,'../data')
    nprocs = torch.cuda.device_count()
    batch_size = int(args.batch_size / nprocs)
    results = []
    exp_nums = len(dataset_feat_net_triples)

    logger.info("-----\nTotal %d experiments in this run:" % exp_nums)
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        logger.info('{}/{} - {} - {} - {}'.format(
            exp_id + 1, exp_nums, dataset_name, feat_str, net))
    '''
    flush()的作用是刷新缓冲区。
    缓冲区的刷新有三种:1,缓冲区满自动刷新；2，flush刷新；3，程序结束自动刷新
    '''
    sys.stdout.flush()

    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        logger.info('-----\n{}/{} - {} - {} - {}'.format(
            exp_id + 1, exp_nums, dataset_name, feat_str, net))
        sys.stdout.flush()
        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)

        model_func = get_model_with_default_configs(net)
        # print(model_func)
        train_acc, acc, std, duration = joint_cl_exp(
            local_rank,
            logger,
            dataset,
            model_func,
            folds=args.n_fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            epoch_select=args.epoch_select,
            with_eval_mode=args.with_eval_mode,
            semi_split=args.semi_split,
            add_mask=args.add_mask)

        summary1 = 'data={}, model={}, feat={}, eval={}'.format(
            dataset_name, net, feat_str, args.epoch_select)
        summary2 = 'train_acc={:.2f}, test_acc={:.2f} ± {:.2f}, sec={}'.format(
            train_acc * 100, acc * 100, std * 100, round(duration, 2))  # round() 方法返回浮点数x的四舍五入值。
        results += ['{}: {}, {}'.format('fin-result', summary1, summary2)]
        logger.info('{}: {}, {}'.format('mid-result', summary1, summary2))
    logger.info('-----\n{}'.format('\n'.join(results)))


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


if __name__ == '__main__':
    set_seed(args.seed)

    device_id = 'cuda:%d' % (args.gpu)
    device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

    args.save = '{}-{}-{}'.format(args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join('exp', args.exp, args.save)
    # create_exp_dir(args.save, glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    # logger.addHandler(fh)
    logger.info(args)


    if args.exp == 'joint_cl_exp':
        run_joint_cl_exp(args, device, logger)
    else:
        raise ValueError('Unknown exp {} to run'.format(args.exp))
