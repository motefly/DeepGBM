import argparse, os, logging, random, time
import numpy as np
import math
import time
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import pdb
import numpy as np
import sys
import glob
import scipy.sparse
import sklearn
from sklearn import preprocessing

from helper import *
from models import *
from train_models import *

parser = argparse.ArgumentParser(description = 'DeepGBM Model and Baselines')
parser.add_argument('-data', type = str, default = 'YAHOO')
parser.add_argument('-model', type = str, default = 'deepgbm')

parser.add_argument('-batch_size', type = int, default = 128)
parser.add_argument('-test_batch_size', type = int, default = 50000)

parser.add_argument('-seed', type = str, default = '1')
parser.add_argument('-log_freq', type = int, default = 100)
parser.add_argument('-test_freq', type = int, default = 1000)

parser.add_argument('-l2_reg', type = float, default = 1e-6)
parser.add_argument('-l2_reg_opt', type = float, default = 5e-4)
parser.add_argument('-plot_title', type = str, default = None)

parser.add_argument('-emb_epoch', type = int, default = 1)
parser.add_argument('-emb_lr', type = float, default = 1e-3)
parser.add_argument('-emb_opt', type = str, default = "Adam")

parser.add_argument('-nslices', type = int, default = 10)
parser.add_argument('-ntrees', type = int, default = 100)

parser.add_argument('-tree_layers', type = str, default = "10,5")
parser.add_argument('-cate_layers', type = str, default = "32,32")

parser.add_argument('-maxleaf', type = int, default = 128)
parser.add_argument('-mindata', type = int, default = 40)
parser.add_argument('-tree_lr', type = float, default = 0.15)
parser.add_argument('-embsize', type = int, default = 20)
parser.add_argument('-cate_embsize', type = int, default = 4)

parser.add_argument('-lr', type = float, default = 1e-3)
parser.add_argument('-opt', type = str, default = 'Adam')

parser.add_argument('-max_epoch', type = int, default = 50)
parser.add_argument('-online_epoch', type = int, default = 1)
parser.add_argument('-online_bz', type = int, default = 128)
parser.add_argument('-loss_init', type = float, default = 1.0)
parser.add_argument('-loss_dr', type = float, default = 0.9)

parser.add_argument('-group_method', type = str, default = 'Random')

parser.add_argument('-feat_per_group', type = int, default = 128)
parser.add_argument('-loss_de', type = int, default = 5)
parser.add_argument('-task', type = str, default = 'regression')
parser.add_argument('-offline', action = 'store_true')


args = parser.parse_args()
assert(args.nslices <= args.ntrees)

plot_title = args.data + "_" + args.opt + "_s" + str(args.seed) + "_ns" + str(args.nslices) + "_nt" + str(args.ntrees)
plot_title += "_lf" + str(args.maxleaf) 
plot_title += "_lr" +str(args.lr) + "_lde" + str(args.loss_de) + "_ldr" + str(args.loss_dr)
plot_title += "_" + args.model
plot_title += "_emb" + str(args.embsize)
plot_title += '_' + args.plot_title
plot_title += '_' + args.group_method
if args.offline:
    plot_title += '_' + 'offline'
else:
    plot_title += '_' + 'online'

args.seeds = [int(x) for x in args.seed.split(',')]
random.seed(args.seeds[0])
np.random.seed(args.seeds[0])
torch.cuda.manual_seed_all(args.seeds[0])

def TrainGBDT2(train_x, train_y, test_x, test_y, lr, num_trees, maxleaf):
    num_class = 1
    if args.task == 'regression':
        objective = "regression"
        metric = "mse"
        boost_from_average = True
    else:
        objective = "binary"
        metric = "auc"
        num_class = 1
        boost_from_average = True
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'num_class': num_class,
        'objective': objective,
        'metric': metric,
        'num_leaves': maxleaf,
        'min_data': 40,
        'boost_from_average': boost_from_average,
        'num_threads': 6,
        'feature_fraction': 0.8,
        'bagging_freq': 3,
        'bagging_fraction': 0.9,
        'learning_rate': lr,
    }
    lgb_train_y = train_y.reshape(-1)
    lgb_test_y = test_y.reshape(-1)
    lgb_train = lgb.Dataset(train_x, lgb_train_y, params=params)
    lgb_eval = lgb.Dataset(test_x, lgb_test_y, reference=lgb_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=num_trees,
                    early_stopping_rounds=20,
                    valid_sets=lgb_eval)
    preds = gbm.predict(test_x)
    preds = preds.astype(np.float32)
    metric = eval_metrics(args.task, lgb_test_y, preds)
    print(metric)
    return gbm

def norm_data(trn_x, vld_x, mean=None, std=None):
    if mean is None:
        mean = np.mean(trn_x, axis=0)
        std = np.std(trn_x, axis=0)
    trn_x = (trn_x - mean) / (std + 1e-5)
    vld_x = (vld_x - mean) / (std + 1e-5)
    return trn_x, vld_x, mean, std

def gbdt_offline():
    root = "data/%s/online"%(args.data+'_num')
    trn_x = np.load(root+"0_train_features.npy")
    trn_y = np.load(root+"0_train_labels.npy")
    vld_x = np.load(root+"1_test_features.npy")
    vld_y = np.load(root+"1_test_labels.npy")
    trn_x = trn_x.astype(np.float32)
    trn_y = trn_y.astype(np.float32)
    vld_x = vld_x.astype(np.float32)
    vld_y = vld_y.astype(np.float32)
    trn_x, vld_x, mean, std = norm_data(trn_x, vld_x)
    maxleaf = args.maxleaf
    gbm = TrainGBDT2(trn_x, trn_y, vld_x, vld_y, args.tree_lr, args.ntrees, maxleaf)
    for t in range(1, 5):
        trn_x = np.load(root+"%d_train_features.npy"%(t))
        trn_y = np.load(root+"%d_train_labels.npy"%(t))
        vld_x = np.load(root+"%d_test_features.npy"%(t+1))
        vld_y = np.load(root+"%d_test_labels.npy"%(t+1))
        trn_x = trn_x.astype(np.float32)
        trn_y = trn_y.astype(np.float32)
        vld_x = vld_x.astype(np.float32)
        vld_y = vld_y.astype(np.float32)
        trn_x, vld_x, _, _ = norm_data(trn_x, vld_x, mean, std)
        preds = gbm.predict(vld_x)
        preds = preds.astype(np.float32)
        # auc = sklearn.metrics.roc_auc_score(vld_y, preds)
        metric = eval_metrics(args.task, vld_y, preds)
        print(metric)
    return gbm

def gbdt_online():
    root = "data/%s/online"%(args.data+'_num')
    trn_x = np.load(root+"0_train_features.npy")
    trn_y = np.load(root+"0_train_labels.npy")
    vld_x = np.load(root+"1_test_features.npy")
    vld_y = np.load(root+"1_test_labels.npy")
    trn_x = trn_x.astype(np.float32)
    trn_y = trn_y.astype(np.float32)
    vld_x = vld_x.astype(np.float32)
    vld_y = vld_y.astype(np.float32)
    trn_x, vld_x, mean, std = norm_data(trn_x, vld_x)
    maxleaf = args.maxleaf
    gbm = TrainGBDT2(trn_x, trn_y, vld_x, vld_y, args.tree_lr, args.ntrees, maxleaf)
    for t in range(1, 5):
        trn_x = np.load(root+"%d_train_features.npy"%(t))
        trn_y = np.load(root+"%d_train_labels.npy"%(t))
        vld_x = np.load(root+"%d_test_features.npy"%(t+1))
        vld_y = np.load(root+"%d_test_labels.npy"%(t+1))
        trn_x = trn_x.astype(np.float32)
        trn_y = trn_y.astype(np.float32).reshape(-1)
        vld_x = vld_x.astype(np.float32)
        vld_y = vld_y.astype(np.float32).reshape(-1)
        trn_x, vld_x, _, _ = norm_data(trn_x, vld_x, mean, std)
        new_gbm = gbm.refit(trn_x, trn_y)
        preds = new_gbm.predict(vld_x)
        preds = preds.astype(np.float32)
        # auc = sklearn.metrics.roc_auc_score(vld_y, preds)
        metric = eval_metrics(args.task, vld_y, preds)
        print(metric)
    return gbm

def GBDT2NN_Refit(train_x, train_y, test_x, test_y, fitted_model, opt, key):
    n_output = train_y.shape[1]
    concate_train_x = np.concatenate([train_x, np.zeros((train_x.shape[0],1), dtype=np.float32)], axis=-1)
    concate_test_x = np.concatenate([test_x, np.zeros((test_x.shape[0],1), dtype=np.float32)], axis=-1)
    if not args.offline:
        TrainWithLog(args, plot_title, concate_train_x, train_y, None,
                     concate_test_x, test_y, fitted_model, opt,
                     args.online_epoch, args.online_bz, n_output, key)
    _,pred_y = EvalTestset(concate_test_x, test_y, fitted_model, args.test_batch_size)
    metric = eval_metrics(args.task, test_y, pred_y)
    print('Final metrics: %s'%str(metric))
    return fitted_model, opt, metric

def gbdt2nn_online():
    root = "data/%s/online"%(args.data+'_num')
    trn_x = np.load(root+"0_train_features.npy")
    trn_y = np.load(root+"0_train_labels.npy")
    vld_x = np.load(root+"1_test_features.npy")
    vld_y = np.load(root+"1_test_labels.npy")
    trn_x = trn_x.astype(np.float32)
    trn_y = trn_y.astype(np.float32)
    vld_x = vld_x.astype(np.float32)
    vld_y = vld_y.astype(np.float32)
    trn_x, vld_x, mean, std = norm_data(trn_x, vld_x)
    num_data = (trn_x, trn_y, vld_x, vld_y)
    fitted_model, opt, metric = train_GBDT2NN(args, num_data, plot_title, key="")
    metrics = [metric]
    for t in range(1, 5):
        trn_x = np.load(root+"%d_train_features.npy"%(t))
        trn_y = np.load(root+"%d_train_labels.npy"%(t))
        vld_x = np.load(root+"%d_test_features.npy"%(t+1))
        vld_y = np.load(root+"%d_test_labels.npy"%(t+1))
        trn_x = trn_x.astype(np.float32)
        trn_y = trn_y.astype(np.float32)
        vld_x = vld_x.astype(np.float32)
        vld_y = vld_y.astype(np.float32)
        trn_x, vld_x, _, _ = norm_data(trn_x, vld_x, mean, std)
        fitted_model, opt, metric = GBDT2NN_Refit(trn_x, trn_y, vld_x, vld_y, fitted_model, opt, key="ol"+str(t))
        metrics.append(metric)
    print(metrics)

def CateNN_refit(train_xs, train_y, test_xs, test_y, fitted_model, opt, key):
    n_output = train_y.shape[1]
    if not args.offline:
        TrainWithLog(args, plot_title, train_xs, train_y, None,
                     test_xs, test_y, fitted_model, opt,
                     args.online_epoch, args.online_bz, n_output, key)
    _,pred_y = EvalTestset(test_xs, test_y, fitted_model, args.test_batch_size)
    metric = eval_metrics(args.task, test_y, pred_y)
    print('Final metrics: %s'%str(metric))
    return fitted_model, opt, metric

def cateNN_online():
    root = "data/%s/online"%(args.data+'_cate')
    train_cate = dh.read_cate_data(root+'0/')
    _,_,feature_sizes = train_cate
    test_cate = dh.read_cate_data(root+'1/')
    cate_data = dh.trans_cate_data((train_cate, test_cate))
    fitted_model, opt, metric = train_cateModels(args, cate_data, plot_title, key="")
    metrics = [metric]
    for t in range(1, 5):
        train_cate = dh.read_cate_data(root+str(t)+'/')
        test_cate = dh.read_cate_data(root+str(t+1)+'/')
        train_cate, test_cate = dh.trans_cate_data((train_cate, test_cate), feature_sizes)
        trn_x,trn_y,_ = train_cate
        vld_x,vld_y,_ = test_cate
        fitted_model, opt, metric = CateNN_refit(trn_x, trn_y, vld_x, vld_y, fitted_model, opt, key="ol"+str(t))
        metrics.append(metric)
    print(metrics)


def deepgbm_online():
    root = "data/%s/online"%(args.data+'_num')
    train_cate = dh.read_cate_data(root.replace('_num', '_cate')+'0/')
    test_cate = dh.read_cate_data(root.replace('_num', '_cate')+'1/')
    cate_data = dh.trans_cate_data((train_cate, test_cate))
    _,_,feature_sizes=train_cate
    trn_x = np.load(root+"0_train_features.npy")
    trn_y = np.load(root+"0_train_labels.npy")
    vld_x = np.load(root+"1_test_features.npy")
    vld_y = np.load(root+"1_test_labels.npy")
    trn_x = trn_x.astype(np.float32)
    trn_y = trn_y.astype(np.float32)
    vld_x = vld_x.astype(np.float32)
    vld_y = vld_y.astype(np.float32)
    trn_x, vld_x, mean, std = norm_data(trn_x, vld_x)
    num_data = (trn_x, trn_y, vld_x, vld_y)
    fitted_model, opt, metric = train_DEEPGBM(args, num_data, cate_data, plot_title, key="")
    metrics = [metric]
    for t in range(1, 5):
        trn_x = np.load(root+"%d_train_features.npy"%(t))
        trn_y = np.load(root+"%d_train_labels.npy"%(t))
        vld_x = np.load(root+"%d_test_features.npy"%(t+1))
        vld_y = np.load(root+"%d_test_labels.npy"%(t+1))
        train_cate = dh.read_cate_data(root.replace('_num', '_cate')+'%d/'%t)
        test_cate = dh.read_cate_data(root.replace('_num', '_cate')+'%d/'%(t+1))
        train_cate, test_cate= dh.trans_cate_data((train_cate, test_cate), feature_sizes)
        trn_x = trn_x.astype(np.float32)
        trn_y = trn_y.astype(np.float32)
        vld_x = vld_x.astype(np.float32)
        vld_y = vld_y.astype(np.float32)
        trn_x, vld_x, _, _ = norm_data(trn_x, vld_x, mean, std)
        fitted_model, opt, metric = DEEPGBM_Refit(trn_x, trn_y, vld_x, vld_y,
                                                train_cate, test_cate,
                                                fitted_model, opt,
                                                key="ol"+str(t))
        metrics.append(metric)
    print(metrics)

def DEEPGBM_Refit(train_x, train_y, test_x, test_y, train_cate,
                  test_cate, fitted_model, opt, key):
    n_output = train_y.shape[1]
    if args.task == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()
    concate_train_x = np.concatenate([train_x, np.zeros((train_x.shape[0],1), dtype=np.float32)], axis=-1)
    concate_test_x = np.concatenate([test_x, np.zeros((test_x.shape[0],1), dtype=np.float32)], axis=-1)
    train_xc, _, _ = train_cate
    test_xc, _, _ = test_cate
    if not args.offline:
        TrainWithLog(args, plot_title, concate_train_x, train_y, None,
                     concate_test_x, test_y, fitted_model, opt,
                     args.online_epoch, args.online_bz, n_output, key,
                     train_x_opt = train_xc, test_x_opt = test_xc)
    _,pred_y = EvalTestset(concate_test_x, test_y, fitted_model, args.test_batch_size, test_xc)
    metric = eval_metrics(args.task, test_y, pred_y)
    print('Final metrics: %s'%str(metric))
    return fitted_model, opt, metric

def D1_Refit(train_x, train_y, test_x, test_y,
                 train_cate, test_cate,
                 fitted_model, opt, gbm, key):
    train_xc, _, _ = train_cate
    test_xc, _, _ = test_cate
    n_output = train_y.shape[1]
    trn_pred = gbm.predict(train_x, raw_score = True)
    tst_pred = gbm.predict(test_x, raw_score = True)
    del train_x, test_x
    if not args.offline:
        TrainWithLog(args, plot_title, trn_pred, train_y, None,
                     tst_pred, test_y, fitted_model, opt,
                     args.online_epoch, args.online_bz, n_output, key,
                     train_x_opt = train_xc, test_x_opt = test_xc)
    _,pred_y = EvalTestset(tst_pred, test_y, fitted_model, args.test_batch_size, test_xc)
    metric = eval_metrics(args.task, test_y, pred_y)
    print('Final metrics: %s'%str(metric))
    return fitted_model, opt, metric

def d1_online():
    root = "data/%s/online"%(args.data+'_num')
    trn_x = np.load(root+"0_train_features.npy")
    trn_y = np.load(root+"0_train_labels.npy")
    vld_x = np.load(root+"1_test_features.npy")
    vld_y = np.load(root+"1_test_labels.npy")
    trn_x = trn_x.astype(np.float32)
    trn_y = trn_y.astype(np.float32)
    vld_x = vld_x.astype(np.float32)
    vld_y = vld_y.astype(np.float32)
    trn_x, vld_x, mean, std = norm_data(trn_x, vld_x)
    num_data = (trn_x, trn_y, vld_x, vld_y)
    train_cate = dh.read_cate_data(root.replace('_num', '_cate')+'0/')
    test_cate = dh.read_cate_data(root.replace('_num', '_cate')+'1/')
    cate_data = dh.trans_cate_data((train_cate, test_cate))
    _,_,feature_sizes=train_cate
    fitted_model, opt, metric, gbm = train_D1(args, num_data, cate_data, plot_title, key="")
    metrics = [metric]
    for t in range(1, 5):
        trn_x = np.load(root+"%d_train_features.npy"%(t))
        trn_y = np.load(root+"%d_train_labels.npy"%(t))
        vld_x = np.load(root+"%d_test_features.npy"%(t+1))
        vld_y = np.load(root+"%d_test_labels.npy"%(t+1))
        trn_x = trn_x.astype(np.float32)
        trn_y = trn_y.astype(np.float32)
        vld_x = vld_x.astype(np.float32)
        vld_y = vld_y.astype(np.float32)
        trn_x, vld_x, _, _ = norm_data(trn_x, vld_x, mean, std)
        train_cate = dh.read_cate_data(root.replace('_num', '_cate')+'%d/'%t)
        test_cate = dh.read_cate_data(root.replace('_num', '_cate')+'%d/'%(t+1))
        train_cate, test_cate= dh.trans_cate_data((train_cate, test_cate), feature_sizes)
        fitted_model, opt, metric = D1_Refit(trn_x, trn_y, vld_x, vld_y,
                                            train_cate, test_cate,
                                              fitted_model, opt, gbm,
                                              key="ol"+str(t))
        metrics.append(metric)
    print(metrics)

def run_all_cate(plot_title):
    root = "data/%s/online"%(args.data+'_num')
    train_cate = dh.read_cate_data(root.replace('_num', '_cate')+'0/')
    test_cate = dh.read_cate_data(root.replace('_num', '_cate')+'1/')
    cate_data = dh.trans_cate_data((train_cate, test_cate))
    _,_,feature_sizes=train_cate
    seeds = args.seeds
    cate_data_bs=[]
    all_res = {'LR':[],'FM':[],'DeepFM':[],'WideNdeep':[],'PNN':[]}
    for t in range(1, 5):
        train_cate = dh.read_cate_data(root.replace('_num', '_cate')+'%d/'%t)
        test_cate = dh.read_cate_data(root.replace('_num', '_cate')+'%d/'%(t+1))
        train_cate, test_cate= dh.trans_cate_data((train_cate, test_cate), feature_sizes)
        cate_data_bs.append((train_cate, test_cate))
    print('Data loaded!')
    
    def run(model, seed):
        local_plot_title = plot_title.replace('cate',model) + str(seed)
        args.model = model
        cate_layers = args.cate_layers
        if model == 'pnn':
            args.cate_layers = cate_layers + ',' + cate_layers.split(',')[-1]
        fitted_model, opt, metric = train_cateModels(args, cate_data, local_plot_title, key="")
        args.cate_layers = cate_layers
        offline_res = [metric]
        online_res = [metric]
        args.offline = True
        for t in range(4):
            train_cate, test_cate = cate_data_bs[t]
            train_xc, train_y, _ = train_cate
            test_xc, test_y, _ = test_cate
            _, opt, metric = CateNN_refit(train_xc, train_y,
                                        test_xc, test_y,
                                        fitted_model, opt,
                                        key="olf"+str(t))
            offline_res.append(metric)
        args.offline = False
        for t in range(4):
            train_cate, test_cate = cate_data_bs[t]      
            train_xc, train_y, _ = train_cate
            test_xc, test_y, _ = test_cate  
            _, opt, metric = CateNN_refit(train_xc, train_y,
                                        test_xc, test_y,
                                        fitted_model, opt,
                                        key="ol"+str(t))
            online_res.append(metric)
        return online_res,offline_res

    for seed in args.seeds:
        args.seeds = [seed]
        max_epoch = args.max_epoch
        all_res['DeepFM'].append(run('deepfm',seed))
        all_res['WideNdeep'].append(run('wideNdeep',seed))
        all_res['PNN'].append(run('pnn',seed))
        args.max_epoch = 1
        all_res['LR'].append(run('lr',seed))
        all_res['FM'].append(run('fm',seed))
        args.seeds = seeds
        args.max_epoch = max_epoch
        print(seed,all_res)

if __name__ == '__main__':
    if args.model == 'gbdt':
        if not args.offline:
            gbdt_online()
        else:
            gbdt_offline()
    elif args.model == 'gbdt2nn':
        gbdt2nn_online()
    elif args.model == 'deepgbm':
        deepgbm_online()
    elif args.model == 'd1':
        d1_online()
    elif args.model == 'all_cate':
        run_all_cate(plot_title)
    else:
        cateNN_online()
