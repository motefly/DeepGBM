import argparse, os, logging, random, time
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter

from models.components import *

import pdb

class PNN(torch.nn.Module):
    def __init__(self, field_size, feature_sizes, embedding_size=4,
                 h_depth = 3, deep_layers=[32, 32, 32], is_deep_dropout=True, dropout_deep=[0.5, 0.5, 0.5], use_inner_product = True, use_outer_product = False,
                 deep_layers_activation='relu', is_batch_norm = True,
                 use_cuda=True, task='regression'
                 ):
        super(PNN, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.use_inner_product = use_inner_product
        self.use_outer_product = use_outer_product
        self.deep_layers_activation = deep_layers_activation
        self.is_batch_norm = is_batch_norm
        self.use_cuda = use_cuda
        self.task = task
        
        """
            check cuda
        """
        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")

        """
            check use inner_product or outer_product
        """
        if self.use_inner_product and self.use_inner_product:
            print("The model uses both inner product and outer product")
        elif self.use_inner_product:
            print("The model uses inner product (IPNN))")
        elif self.use_outer_product:
            print("The model uses outer product (OPNN)")
        else:
            print("The model is sample deep model only! Neither inner product or outer product is used")

        """
            embbedding part
        """
        print("Init embeddings")
        self.embedding = nn.Embedding(sum(self.feature_sizes), self.embedding_size)
        print("Init embeddings finished")

        """
            first order part (linear part)
        """
        print("Init first order part")
        self.first_weight = Parameter(torch.Tensor(self.embedding_size, self.deep_layers[0]))
        stdv = math.sqrt(6.0 /(self.embedding_size + self.deep_layers[0]))
        self.first_weight.data.uniform_(-stdv, stdv)
        
        self.bias = torch.nn.Parameter(torch.randn(self.deep_layers[0]), requires_grad=True)
        print("Init first order part finished")

        """
            second order part (quadratic part)
        """
        print("Init second order part")
        if self.use_inner_product:
            self.inner_weight = Parameter(torch.Tensor(self.embedding_size, self.deep_layers[0]))
            self.inner_weight.data.uniform_(-stdv, stdv)

        if self.use_outer_product:
            self.outer_conv = nn.Conv2d(1,self.deep_layers[0],self.embedding_size,bias=False)
        print("Init second order part finished")


        print("Init nn part")

        for i, h in enumerate(self.deep_layers[1:], 1):
            setattr(self, 'linear_' + str(i), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            if self.is_batch_norm:
                setattr(self, 'batch_norm_' + str(i), nn.BatchNorm1d(deep_layers[i]))
            if self.is_deep_dropout:
                setattr(self, 'linear_' + str(i) + '_dropout', nn.Dropout(self.dropout_deep[i]))
        self.deep_last_layer = nn.Linear(self.deep_layers[-1], 1)
        print("Init nn part succeed")

                
        if self.task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()

        print("Init succeed")

    def batchmul(self, x, models, embed_w, length):
        out = one_hot(x, length)
        out = out.view(x.size(0), models, -1)
        out = out.transpose(0, 1).contiguous()
        out = torch.bmm(out, embed_w)
        out = out.transpose(0, 1).contiguous()
        out = out.view(x.size(0), -1)
        return out

    def forward(self, X):
        """
        :param Xi: index input tensor, batch_size * k * 1
        :param Xv: value input tensor, batch_size * k * 1
        :param is_pretrain: the para to decide fm pretrain or not
        :return: the last output
        """

        """
            embedding
        """
        Xi = X.long()
        emb = self.embedding(Xi.view(X.size(0)*self.field_size)).view(X.size(0),self.field_size, self.embedding_size)
        """
            first order part (linear part)
        """
        first_order = torch.mm(emb.view(-1, self.embedding_size), self.first_weight).view(X.size(0), self.field_size, self.deep_layers[0])
        first_order = torch.sum(first_order, 1)
        """
            second order part (quadratic part)
        """
        if self.use_inner_product:
            inner_product = torch.mm(emb.view(-1, self.embedding_size), self.inner_weight).view(X.size(0), self.field_size, self.deep_layers[0])
            inner_product = torch.sum(inner_product * inner_product, 1)
            first_order = first_order + inner_product

        if self.use_outer_product:
            emb_sum = torch.sum(emb, 1)
            emb_matrix = torch.bmm(emb_sum.view([-1,self.embedding_size,1]),emb_sum.view([-1,1,self.embedding_size])).view(-1,1,self.embedding_size,self.embedding_size)
            outer_product = self.outer_conv(emb_matrix).view(-1,self.deep_layers[0])
            first_order = first_order + outer_product

        """
            nn part
        """
        if self.deep_layers_activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif self.deep_layers_activation == 'tanh':
            activation = nn.Tanh()
        else:
            activation = nn.ReLU()
        x_deep = first_order
        for i, h in enumerate(self.deep_layers[1:], 1):
            x_deep = getattr(self, 'linear_' + str(i))(x_deep)
            if self.is_batch_norm:
                x_deep = getattr(self, 'batch_norm_' + str(i))(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = getattr(self, 'linear_' + str(i) + '_dropout')(x_deep)
        x_deep = self.deep_last_layer(x_deep)
        out = torch.sum(x_deep, 1)
        if self.task != 'regression':
            return nn.Sigmoid()(out)
        return out
    
    def true_loss(self, out, target):
        return self.criterion(out.view(-1), target.view(-1))
