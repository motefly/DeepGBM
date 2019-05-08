import argparse, os, logging, random, time, math
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter

from models.components import *

import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    type_prefix = torch.cuda
else:
    type_prefix = torch

class DeepFM(torch.nn.Module):
    def __init__(self, field_size, feature_sizes, embedding_size = 4,
                 is_shallow_dropout = True, dropout_shallow = [0.5,0.5],
                 h_depth = 2, deep_layers = [32, 32], is_deep_dropout = False,
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation = 'relu',
                 is_batch_norm = True, use_wide = False,
                 use_fm = True, use_deep = True,
                 use_cuda = True, task = 'binary'):
        super(DeepFM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.is_batch_norm = is_batch_norm
        self.use_fm = use_fm
        self.use_wide = use_wide
        self.use_deep = use_deep
        self.use_cuda = use_cuda
        self.task = task
        """
            check cuda
        """
        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")
        """
            check use fm
        """
        if self.use_fm and self.use_deep:
            print("The model is deepfm(fm+deep layers)")
        elif self.use_wide and self.use_deep:
            print("The model is wide&deep(lr+deep layers)")
        elif self.use_fm:
            print("The model is fm only")
        elif self.use_deep:
            print("The model is deep layers only")
        elif self.use_wide:
            print("The model is lr only")
        else:
            print("You have to choose more than one of (fm, deep) models to use")
            exit(1)
        """
            bias
        """
        if self.use_fm or self.use_wide:
            self.bias = torch.nn.Parameter(torch.randn(1))
        """
            fm part
        """
        stdv = math.sqrt(1.0 /len(self.feature_sizes))
        if self.use_fm:
            print("Init fm part")
            # new embed
            self.fm_first_order_embedding = nn.Embedding(sum(self.feature_sizes), 1)
            self.fm_first_order_embedding.weight.data.normal_(0,stdv)
            if self.dropout_shallow:
                self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            # new embed
            self.fm_second_order_embedding = nn.Embedding(sum(self.feature_sizes), self.embedding_size)
            self.fm_second_order_embedding.weight.data.normal_(0,stdv)
            
            if self.dropout_shallow:
                self.fm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])
            print("Init fm part succeed")

        """
            wide part
        """
        if self.use_wide:
            print("Init wide/lr part")
            # new embed
            self.fm_first_order_embedding = nn.Embedding(sum(self.feature_sizes), 1)
            self.fm_first_order_embedding.weight.data.normal_(0,stdv)
            if self.dropout_shallow:
                self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            print("Init wide/lr part succeed")

        """
            deep part
        """
        if self.use_deep:
            print("Init deep part")
            if not self.use_fm:
                # new embed
                self.fm_second_order_embedding = nn.Embedding(sum(self.feature_sizes), self.embedding_size)
                self.fm_second_order_embedding.weight.data.normal_(0,stdv)

            if self.is_deep_dropout:
                self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])

            self.linear_1 = nn.Linear(self.field_size*self.embedding_size,deep_layers[0])
            if self.is_batch_norm:
                self.batch_norm_1 = nn.BatchNorm1d(deep_layers[0])
            if self.is_deep_dropout:
                self.linear_1_dropout = nn.Dropout(self.dropout_deep[1])
            for i, h in enumerate(self.deep_layers[1:], 1):
                setattr(self,'linear_'+str(i+1), nn.Linear(self.deep_layers[i-1], self.deep_layers[i]))
                if self.is_batch_norm:
                    setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(deep_layers[i]))
                if self.is_deep_dropout:
                    setattr(self, 'linear_'+str(i+1)+'_dropout', nn.Dropout(self.dropout_deep[i+1]))

            print("Init deep part succeed")
        
        if self.task == 'regression':
            self.criterion = nn.MSELoss()
            # self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.BCELoss()

        print("Init succeed")
        
    def forward(self, X):
        """
        :param Xi_train: index input tensor, batch_size * k * 1
        :param Xv_train: value input tensor, batch_size * k * 1
        :return: the last output
        """
        """
            fm part
        """
        Xi = X.long()
        if self.use_fm:
            fm_first_order = self.fm_first_order_embedding(Xi.view(X.size(0)*self.field_size)).view(X.size(0), -1)
            if self.is_shallow_dropout:
                fm_first_order = self.fm_first_order_dropout(fm_first_order)

            # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
            fm_second_order_emb = self.fm_second_order_embedding(Xi.view(X.size(0)*self.field_size)).view(X.size(0), self.field_size, -1)
            fm_sum_second_order_emb = torch.sum(fm_second_order_emb, 1)
            fm_sum_second_order_emb_square = fm_sum_second_order_emb*fm_sum_second_order_emb # (x+y)^2
            fm_second_order_emb_square = fm_second_order_emb * fm_second_order_emb
            fm_second_order_emb_square_sum = torch.sum(fm_second_order_emb_square, 1) #x^2+y^2
            fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
            if self.is_shallow_dropout:
                fm_second_order = self.fm_second_order_dropout(fm_second_order)
        """
            wide part
        """
        if self.use_wide:
            fm_first_order = self.fm_first_order_embedding(Xi.view(X.size(0)*self.field_size)).view(X.size(0), -1)
            if self.is_shallow_dropout:
                fm_first_order = self.fm_first_order_dropout(fm_first_order)
        """
            deep part
        """
        if self.use_deep:
            if self.use_fm:
                deep_emb = fm_second_order_emb.reshape(Xi.size(0),-1)
            else:
                deep_emb = self.fm_second_order_embedding(Xi.view(X.size(0)*self.field_size)).view(X.size(0), -1)
            if self.deep_layers_activation == 'sigmoid':
                activation = nn.Sigmoid()
            elif self.deep_layers_activation == 'tanh':
                activation = nn.Tanh()
            else:
                activation = nn.ReLU()
            if self.is_deep_dropout:
                deep_emb = self.linear_0_dropout(deep_emb)
            x_deep = self.linear_1(deep_emb)
            if self.is_batch_norm:
                x_deep = self.batch_norm_1(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = self.linear_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
                if self.is_batch_norm:
                    x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
                x_deep = activation(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)
        """
            sum
        """
        if self.use_fm and self.use_deep:
            total_sum = torch.sum(fm_first_order,1) + torch.sum(fm_second_order,1) + torch.sum(x_deep,1) + self.bias
        elif self.use_fm:
            total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + self.bias
        elif self.use_wide and self.use_deep:
            total_sum = torch.sum(fm_first_order, 1) + torch.sum(x_deep,1) + self.bias
        elif self.use_wide:
            total_sum = torch.sum(fm_first_order, 1) + self.bias
        else:
            total_sum = torch.sum(x_deep,1)
        if self.task != 'regression':
            return nn.Sigmoid()(total_sum)
        return total_sum

    def true_loss(self, out, target):
        return self.criterion(out.view(-1), target.view(-1))
