import argparse, os, logging, random, time
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter

from models.deepfm import DeepFM
from models.gbdt2nn import GBDT2NN
from models.components import Dense

import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    type_prefix = torch.cuda
else:
    type_prefix = torch

class DeepGBM(torch.nn.Module):
    def __init__(self, nume_input_size=None, used_features=None,
                 tree_layers=None, output_w=None, output_b=None, task=None,
                 cate_field_size=None, feature_sizes=None, embedding_size = 4,
                 is_shallow_dropout = True, dropout_shallow = [0.5,0.5],
                 h_depth = 2, deep_layers = [32, 32], is_deep_dropout = False,
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation = 'relu',
                 is_batch_norm = False,
                 num_model = 'gbdt2nn',
                 func = None, maxleaf = 128):
        super(DeepGBM, self).__init__()
        self.num_model = num_model
        if num_model == 'gbdt2nn':
            self.gbdt2nn = GBDT2NN(nume_input_size, used_features,
                                   tree_layers, output_w, output_b, task='regression')
        elif num_model == 'gbdt':
            self.gbdt2nn = None
        self.deepfm = DeepFM(cate_field_size, feature_sizes, embedding_size,
                              is_shallow_dropout, dropout_shallow, 
                              h_depth, deep_layers, is_deep_dropout,
                              dropout_deep, deep_layers_activation,
                              is_batch_norm, task='regression')
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.task = task
        print('Init DeepGBM succeed!')
        if self.task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()

    def forward(self, Xg, Xd):
        Xd = Xd.long()
        if self.num_model == 'gbdt2nn':
            gbdt2nn_out, gbdt2nn_pred = self.gbdt2nn(Xg)
        elif self.num_model == 'gbdt':
            gbdt2nn_out = Xg.float()
            gbdt2nn_pred = None
        else:
            gbdt2nn_out = self.gbdt2nn(Xg)
            gbdt2nn_pred = None
        deepfm_out = self.deepfm(Xd)

        if self.num_model != 'gbdt2nn':
            alpha = self.alpha+0.5
            beta = self.beta+0.5
        else:
            alpha = self.alpha+1
            beta = self.beta
        out = alpha * gbdt2nn_out.view(-1) + beta * deepfm_out.view(-1)
        if self.task != 'regression':
            return nn.Sigmoid()(out), gbdt2nn_pred
        return out, gbdt2nn_pred

    def joint_loss(self, out, target, gbdt2nn_emb_pred, gbdt2nn_emb_target, ratio):
        return (1-ratio) * self.criterion(out.view(-1), target.view(-1)) + ratio * self.gbdt2nn.emb_loss(gbdt2nn_emb_pred, gbdt2nn_emb_target)

    def true_loss(self, out, target):
        return self.criterion(out.view(-1), target.view(-1))
