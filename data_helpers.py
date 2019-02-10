import pandas as pd
import pdb
import numpy as np
import sys
import logging
import glob
from sklearn import preprocessing
from math import *

logging.basicConfig(level = logging.INFO, format = '%(asctime)s [INFO] %(message)s')
# root = "/home/zhenhui/Data/"
root = 'data/'

def load_data(data):
    if 'offline' in data or 'online' in data:
        data_dir = root#'../Online-Prediction-Benchmarks/data'
        if 'cate' in data:
            train = read_cate_data(data_dir+'/%s/train/'%data)
            test = read_cate_data(data_dir+'/%s/test/'%data)
            logging.info('Categorical data loaded.\n train_x shape: {trn_x_shape}. train_y shape: {trn_y_shape}.\n test_x shape: {vld_x_shape}. test_y shape: {vld_y_shape}.' .format(trn_x_shape = train[0].shape, trn_y_shape = train[1].shape, vld_x_shape = test[0].shape, vld_y_shape = test[1].shape))
            return train, test
        trn_x = np.load(data_dir+"/%s/train_features.npy"%data).astype(np.float32)
        trn_y = np.load(data_dir+"/%s/train_labels.npy"%data).astype(np.float32)
        vld_x = np.load(data_dir+"/%s/test_features.npy"%data).astype(np.float32)
        vld_y = np.load(data_dir+"/%s/test_labels.npy"%data).astype(np.float32)
        mean = np.mean(trn_x, axis=0)
        std = np.std(trn_x, axis=0)
        trn_x = (trn_x - mean) / (std + 1e-5)
        vld_x = (vld_x - mean) / (std + 1e-5)
        logging.info('data loaded.\n train_x shape: {trn_x_shape}. train_y shape: {trn_y_shape}.\n test_x shape: {vld_x_shape}. test_y shape: {vld_y_shape}.' .format(trn_x_shape = trn_x.shape, trn_y_shape = trn_y.shape, vld_x_shape = vld_x.shape, vld_y_shape = vld_y.shape))
    return trn_x, trn_y, vld_x, vld_y

def read_cate_data(dir_path):
    y = np.load(dir_path + '_label.npy')[:,None]
    xi = np.load(dir_path + '_index.npy')
    feature_sizes = np.load(dir_path + '_feature_sizes.npy').tolist()
    print("loaded from %s."%dir_path)
    # xv = np.load(dir_path + '_value.npy')
    # x = np.concatenate([xi[:,:,None],xv[:,:,None]], axis=-1)
    return xi, y.astype(np.float32), feature_sizes

# for fast version cateNN
def trans_cate_data(cate_data, old_feature_sizes=None):
    train, test = cate_data
    train_xs, train_y, feature_sizes = train
    test_xs, test_y, _ = test
    if old_feature_sizes is not None:
        feature_sizes = old_feature_sizes
    sum_feats = feature_sizes[0]
    for idx in range(1, len(feature_sizes)):
        # sum_feats.append(sum_feats[idx-1]+feature_sizes[idx-1])
        train_xs[:,idx] += sum_feats
        test_xs[:,idx] += sum_feats
        sum_feats += feature_sizes[idx]
    return ((train_xs, train_y, feature_sizes), (test_xs, test_y, _))
