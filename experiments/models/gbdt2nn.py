import argparse, os, logging, random, time
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from models.components import *

import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    type_prefix = torch.cuda
else:
    type_prefix = torch
    
class GBDT2NN(nn.Module):
    def __init__(self, input_size, used_features,
                 tree_layers, output_w, output_b, task):
        super(GBDT2NN, self).__init__()
        print('Init GBDT2NN')
        self.task = task
        self.n_models = len(used_features)
        self.tree_layers = tree_layers
        n_feature = len(used_features[0])
        used_features = np.asarray(used_features).reshape(-1)
        self.used_features = Variable(torch.from_numpy(used_features).to(device), requires_grad=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        assert len(tree_layers) > 0
        self.bdenses = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bdenses.append(BatchDense(self.n_models, n_feature, tree_layers[0]))
        for i in range(1, len(tree_layers)):
            self.bdenses.append(BatchDense(self.n_models, tree_layers[i-1], tree_layers[i]))
        for i in range(len(tree_layers)-1):
            self.bns.append(nn.BatchNorm1d(tree_layers[i] * self.n_models))
        self.out_weight = Variable(torch.from_numpy(output_w).to(device), requires_grad=False)
        self.out_bias = Variable(torch.from_numpy(output_b).to(device), requires_grad=False)
        print('Init GBDT2NN succeed!')
        if self.task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()

    def batchmul(self, x, f):
        out = x.view(x.size(0), self.n_models, -1)
        out = f(out)
        out = out.view(x.size(0), -1)
        return out
    def lastlayer(self, x):
        out = torch.index_select(x, dim=1, index=self.used_features)
        for i in range(len(self.bdenses) - 1):
            out = self.batchmul(out, self.bdenses[i])
            out = self.bns[i](out)
            out = self.relu(out)
        return out
    
    def forward(self, x):
        out = self.lastlayer(x.float())
        pred = self.batchmul(out, self.bdenses[-1])
        out = torch.addmm(self.out_bias, pred, self.out_weight)
        if self.task != 'regression':
            return self.sigmoid(out), pred
        return out, pred

    def emb_loss(self, emb_pred, emb_target):
        loss_weight = torch.abs(torch.sum(self.out_weight, 1))
        l2_loss = nn.MSELoss(reduction='none')(emb_pred, emb_target)*loss_weight
        return torch.mean(torch.sum(l2_loss, dim=1))

    def joint_loss(self, out, target, emb_pred, emb_target, ratio):
        return (1-ratio) * self.criterion(out, target) + ratio * self.emb_loss(emb_pred, emb_target)

    def true_loss(self, out, target):
        return self.criterion(out, target)
