import argparse, os, logging, random, time
import data_helpers as dh
import numpy as np
import math
import time
import sklearn.metrics
import scipy.sparse
import lightgbm as lgb
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tensorboardX as tbx
from sklearn.utils.extmath import softmax

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from tree_model_interpreter import *

import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    type_prefix = torch.cuda
else:
    type_prefix = torch

def GetSplitFeature(d):
    fea_key = "split_feature"
    thr_key = "threshold"
    gain_key = "split_gain"
    features = {}
    if fea_key in d:
        feature = d[fea_key]
        threshold = d[thr_key]
        gain = d[gain_key]
        # gain  = 1
        features[feature] = gain
    for key in d:
        if isinstance(d[key], dict):
            tf = GetSplitFeature(d[key])
            if tf:
                for key in tf:
                    if key not in features:
                        features[key] = 0
                    features[key] += tf[key]
    return features

def TrainGBDT(train_x, train_y, test_x, test_y, lr, num_trees, maxleaf, mindata, task):
    num_class = 1
    if task == 'regression':
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
    preds = gbm.predict(train_x, raw_score=True)
    preds = preds.astype(np.float32)
    return gbm, preds


def SubGBDTLeaf_cls(train_x, test_x, gbm, maxleaf, num_slices, args):
    MAX=train_x.shape[1]
    leaf_preds = gbm.predict(train_x, pred_leaf=True).reshape(train_x.shape[0], -1)
    test_leaf_preds = gbm.predict(test_x, pred_leaf=True).reshape(test_x.shape[0], -1)
    n_trees = leaf_preds.shape[1]
    step = int((n_trees + num_slices - 1) // num_slices)
    step = max(step, 1)
    leaf_output = np.zeros([n_trees, maxleaf], dtype=np.float32)
    for tid in range(n_trees):
        num_leaf = np.max(leaf_preds[:,tid]) + 1
        for lid in range(num_leaf):
            leaf_output[tid][lid] = gbm.get_leaf_output(tid, lid)
    rest_nt = n_trees
    modelI = ModelInterpreter(gbm, args)
    if args.group_method == 'Equal' or args.group_method == 'Random':
        clusterIdx = modelI.EqualGroup(num_slices, args)
        n_feature = args.feat_per_group
    treeI = modelI.trees
    # rand = (args.group_method == 'Random')
    Allset = set([i for i in range(MAX)])
    for n_idx in range(num_slices):
        tree_indices = np.where(clusterIdx == n_idx)[0]
        trees = {}
        tid = 0
        for jdx in tree_indices:
            trees[str(tid)] = treeI[jdx].raw
            tid += 1
        tree_num = len(tree_indices)
        layer_num = 1
        xi = []
        xi_fea = set()
        all_hav = {} # set([i for i in range(MAX)])
        for jdx, tree in enumerate(tree_indices):
            for kdx, f in enumerate(treeI[tree].feature):
                if f == -2:
                    continue
                if f not in all_hav:
                    all_hav[f] = 0
                all_hav[f] += treeI[tree].gain[kdx]
        used_features = []
        rest_feature = []
        all_hav = sorted(all_hav.items(), key=lambda kv: -kv[1])
        used_features = [item[0] for item in all_hav[:n_feature]]
        # if rand:
        # used_features = np.random.choice(MAX, len(used_features), replace = False).tolist()
        used_features_set = set(used_features)
        for kdx in range(max(0, n_feature - len(used_features))):
            used_features.append(MAX)
        cur_leaf_preds = leaf_preds[:, tree_indices]
        cur_test_leaf_preds = test_leaf_preds[:, tree_indices]
        new_train_y = np.zeros(train_x.shape[0])
        for jdx in tree_indices:
            new_train_y += np.take(leaf_output[jdx,:].reshape(-1), leaf_preds[:,jdx].reshape(-1))
        new_train_y = new_train_y.reshape(-1,1).astype(np.float32)
        yield used_features, new_train_y, cur_leaf_preds, cur_test_leaf_preds, np.mean(np.take(leaf_output, tree_indices,0)), np.mean(leaf_output)

class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias_init=0, task='regresion', func=None, length=None):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters(bias_init)
        self.task = task
        self.func = func
        self.length = length
        if task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()
    def reset_parameters(self, bias_init=0):
        stdv = math.sqrt(6.0 /(self.in_features + self.out_features))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(bias_init)
    def forward(self, x):
        if self.func == 'lr':
            x = one_hot(x.long(), self.length)
        out = torch.addmm(self.bias, x.float(), self.weight)
        if self.task != 'regression':
            return nn.Sigmoid()(out)
        return out
    def true_loss(self, out, target):
        return self.criterion(out, target)

class BatchDense(nn.Module):
    def __init__(self, batch, in_features, out_features, bias_init=None):
        super(BatchDense, self).__init__()
        self.batch = batch
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(batch, in_features, out_features))
        self.bias = Parameter(torch.Tensor(batch, 1, out_features))
        self.reset_parameters(bias_init)
    def reset_parameters(self, bias_init=None):
        stdv = math.sqrt(6.0 /(self.in_features + self.out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if bias_init is not None:
            self.bias.data = torch.from_numpy(bias_init)
        else:
            self.bias.data.fill_(0)
    def forward(self, x):
        size = x.size()
        # Todo: avoid the swap axis
        x = x.view(x.size(0), self.batch, -1)
        out = x.transpose(0, 1).contiguous()
        out = torch.baddbmm(self.bias, out, self.weight)
        out = out.transpose(0, 1).contiguous()
        out = out.view(x.size(0), -1)
        return out

def one_hot(y, numslot, mask=None):
    y_tensor = y.type(type_prefix.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], numslot, device=device, dtype=torch.float32, requires_grad=False).scatter_(1, y_tensor, 1)
    if mask is not None:
        y_one_hot = y_one_hot * mask
    y_one_hot = y_one_hot.view(y.shape[0], -1)
    return y_one_hot

def SotfCE(outputs, target):
    log_likelihood = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = torch.mean(-torch.sum(log_likelihood* target, dim=1))
    return loss

class EmbeddingModel(nn.Module):
    def __init__(self, n_models, max_ntree_per_split, embsize, maxleaf, n_output, out_bias=None, task='regression'):
        super(EmbeddingModel, self).__init__()
        self.task = task
        self.n_models = n_models
        self.maxleaf = maxleaf
        self.fcs = nn.ModuleList()
        self.max_ntree_per_split = max_ntree_per_split

        self.embed_w = Parameter(torch.Tensor(n_models, max_ntree_per_split*maxleaf, embsize))
        # torch.nn.init.xavier_normal(self.embed_w)
        stdv = math.sqrt(1.0 /(max_ntree_per_split))
        self.embed_w.data.normal_(0,stdv) # .uniform_(-stdv, stdv)
        
        self.bout = BatchDense(n_models, embsize, 1, out_bias)
        self.bn = nn.BatchNorm1d(embsize * n_models)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.output_fc = Dense(n_models * embsize, n_output)
        self.dropout = torch.nn.Dropout()
        if task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()

    def batchmul(self, x, models, embed_w, length):
        out = one_hot(x, length)
        out = out.view(x.size(0), models, -1)
        out = out.transpose(0, 1).contiguous()
        out = torch.bmm(out, embed_w)
        out = out.transpose(0, 1).contiguous()
        out = out.view(x.size(0), -1)
        return out
        
    def lastlayer(self, x):
        out = self.batchmul(x, self.n_models, self.embed_w, self.maxleaf)
        out = self.bn(out)
        # out = self.tanh(out)
        # out = out.view(x.size(0), self.n_models, -1)
        return out
    def forward(self, x):
        out = self.lastlayer(x)
        out = self.dropout(out)
        out = out.view(x.size(0), self.n_models, -1)
        out = self.bout(out)
        # out = self.output_fc(out)
        sum_out = torch.sum(out,-1,True)
        if self.task != 'regression':
            return self.sigmoid(sum_out), out
        return sum_out, out
    
    def joint_loss(self, out, target, out_inner, target_inner, *args):
        return nn.MSELoss()(out_inner, target_inner)

    def true_loss(self, out, target):
        return self.criterion(out, target)
