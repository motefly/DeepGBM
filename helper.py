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
# import tensorboardX as tbx
# import tensorflow as tf
from sklearn.utils.extmath import softmax

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

import pdb

# summaryPath = "/share_data/deepgbm/0201_sum/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    type_prefix = torch.cuda
else:
    type_prefix = torch

class AdamW(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, model_decay_opt=None,
                 weight_decay_opt=None, key_opt=''):
        super(AdamW, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0, amsgrad=amsgrad)
        self.weight_decay = weight_decay
        self.weight_decay_opt = weight_decay_opt
        self.model_decay_opt = model_decay_opt
        self.key_opt = key_opt
    def step(self, closure=None):
        loss = super(AdamW, self).step(closure)
        for group in self.param_groups:
            if self.weight_decay != 0:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.data.add_(-self.weight_decay, p.data)
        if self.model_decay_opt is not None:
            for name, parameter in self.model_decay_opt.named_parameters():
                if self.key_opt in name:
                    if parameter.grad is not None:
                        parameter.data.add_(-self.weight_decay_opt, parameter.data)
        return loss

def eval_metrics(task, true, pred):
    if task == 'binary':
        logloss = sklearn.metrics.log_loss(true.astype(np.float64), pred.astype(np.float64))
        auc = sklearn.metrics.roc_auc_score(true, pred)
        # error = 1-sklearn.metrics.accuracy_score(true,(pred+0.5).astype(np.int32))
        return (logloss, auc)#, error)
    else:
        mseloss = sklearn.metrics.mean_squared_error(true, pred)
        return mseloss

def EvalTestset(test_x, test_y, model, test_batch_size, test_x_opt=None):
    test_len = test_x.shape[0]
    test_num_batch = math.ceil(test_len / test_batch_size)
    sum_loss = 0.0
    y_preds = []
    model.eval()
    with torch.no_grad():
        for jdx in range(test_num_batch):
            tst_st = jdx * test_batch_size
            tst_ed = min(test_len, tst_st + test_batch_size)
            inputs = torch.from_numpy(test_x[tst_st:tst_ed].astype(np.float32)).to(device)
            if test_x_opt is not None:
                inputs_opt = torch.from_numpy(test_x_opt[tst_st:tst_ed].astype(np.float32)).to(device)
                outputs = model(inputs, inputs_opt)
            else:
                outputs = model(inputs)
            targets = torch.from_numpy(test_y[tst_st:tst_ed]).to(device)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            y_preds.append(outputs)
            loss_tst = model.true_loss(outputs, targets).item()
            sum_loss += (tst_ed - tst_st) * loss_tst
    return sum_loss / test_len, np.concatenate(y_preds, 0)

def TrainWithLog(args, plot_title, train_x, train_y,
                 train_y_opt, test_x, test_y, model, opt,
                 epoch, batch_size, n_output, key="",
                 train_x_opt=None, test_x_opt=None):
    # trn_writer = tf.summary.FileWriter(summaryPath+plot_title+key+"_output/train")
    # tst_writer = tf.summary.FileWriter(summaryPath+plot_title+key+"_output/test")
    if isinstance(test_x, scipy.sparse.csr_matrix):
        test_x = test_x.todense()
    train_len = train_x.shape[0]
    global_iter = 0
    trn_batch_size = batch_size
    train_num_batch = math.ceil(train_len / trn_batch_size)
    total_iterations = epoch * train_num_batch
    start_time = time.time()
    total_time = 0.0
    min_loss = float("Inf")
    # min_error = float("Inf")
    max_auc = 0.0
    for epoch in range(epoch):
        shuffled_indices = np.random.permutation(np.arange(train_x.shape[0]))
        Loss_trn_epoch = 0.0
        Loss_trn_log = 0.0
        log_st = 0
        for local_iter in range(train_num_batch):
            trn_st = local_iter * trn_batch_size
            trn_ed = min(train_len, trn_st + trn_batch_size)
            batch_trn_x = train_x[shuffled_indices[trn_st:trn_ed]]
            if isinstance(batch_trn_x, scipy.sparse.csr_matrix):
                batch_trn_x = batch_trn_x.todense()
            inputs = torch.from_numpy(batch_trn_x.astype(np.float32)).to(device)
            targets = torch.from_numpy(train_y[shuffled_indices[trn_st:trn_ed],:]).to(device)
            model.train()
            if train_x_opt is not None:
                inputs_opt = torch.from_numpy(train_x_opt[shuffled_indices[trn_st:trn_ed]].astype(np.float32)).to(device)
                outputs = model(inputs, inputs_opt)
            else:
                outputs = model(inputs)
            opt.zero_grad()
            if isinstance(outputs, tuple) and train_y_opt is not None:
                # targets_inner = torch.from_numpy(s_train_y_opt[trn_st:trn_ed,:]).to(device)
                targets_inner = torch.from_numpy(train_y_opt[shuffled_indices[trn_st:trn_ed],:]).to(device)
                loss_ratio = args.loss_init * max(0.3,args.loss_dr ** (epoch // args.loss_de))#max(0.5, args.loss_dr ** (epoch // args.loss_de))
                if len(outputs) == 3:
                    loss_val = model.joint_loss(outputs[0], targets, outputs[1], targets_inner, loss_ratio, outputs[2])
                else:
                    loss_val = model.joint_loss(outputs[0], targets, outputs[1], targets_inner, loss_ratio)
                loss_val.backward()
                loss_val = model.true_loss(outputs[0], targets)
            elif isinstance(outputs, tuple):
                loss_val = model.true_loss(outputs[0], targets)
                loss_val.backward()
            else:
                loss_val = model.true_loss(outputs, targets)
                loss_val.backward()
            opt.step()
            loss_val = loss_val.item()
            global_iter += 1
            Loss_trn_epoch += (trn_ed - trn_st) * loss_val
            Loss_trn_log += (trn_ed - trn_st) * loss_val
            if global_iter % args.log_freq == 0:
                print(key+"Epoch-{:0>3d} {:>5d} Batches, Step {:>6d}, Training Loss: {:>9.6f} (AllAvg {:>9.6f})"
                            .format(epoch, local_iter + 1, global_iter, Loss_trn_log/(trn_ed-log_st), Loss_trn_epoch/trn_ed))
                
                # trn_summ = tf.Summary()
                # trn_summ.value.add(tag=args.data+ "/Train/Loss", simple_value = Loss_trn_log/(trn_ed-log_st))
                # trn_writer.add_summary(trn_summ, global_iter)
                log_st = trn_ed
                Loss_trn_log = 0.0
            if global_iter % args.test_freq == 0 or local_iter == train_num_batch - 1:
                if args.model == 'deepgbm' or args.model == 'd1':
                    try:
                        print('Alpha: '+str(model.alpha))
                        print('Beta: '+str(model.beta))
                    except:
                        pass
                # tst_summ = tf.Summary()
                torch.cuda.empty_cache()
                test_loss, pred_y = EvalTestset(test_x, test_y, model, args.test_batch_size, test_x_opt)
                current_used_time = time.time() - start_time
                start_time = time.time()
                total_time += current_used_time
                remaining_time = (total_iterations - (global_iter) ) * (total_time / (global_iter))
                if args.task == 'binary':
                    metrics = eval_metrics(args.task, test_y, pred_y)
                    _, test_auc = metrics
                    # min_error = min(min_error, test_error)
                    max_auc = max(max_auc, test_auc)
                    # tst_summ.value.add(tag=args.data+"/Test/Eval/Error", simple_value = test_error)
                    # tst_summ.value.add(tag=args.data+"/Test/Eval/AUC", simple_value = test_auc)
                    # tst_summ.value.add(tag=args.data+"/Test/Eval/Min_Error", simple_value = min_error)
                    # tst_summ.value.add(tag=args.data+"/Test/Eval/Max_AUC", simple_value = max_auc)
                    print(key+"Evaluate Result:\nEpoch-{:0>3d} {:>5d} Batches, Step {:>6d}, Testing Loss: {:>9.6f}, Testing AUC: {:8.6f}, Used Time: {:>5.1f}m, Remaining Time: {:5.1f}m"
                            .format(epoch, local_iter + 1, global_iter, test_loss, test_auc, total_time/60.0, remaining_time/60.0))
                else:
                    print(key+"Evaluate Result:\nEpoch-{:0>3d} {:>5d} Batches, Step {:>6d}, Testing Loss: {:>9.6f}, Used Time: {:>5.1f}m, Remaining Time: {:5.1f}m"
                            .format(epoch, local_iter + 1, global_iter, test_loss, total_time/60.0, remaining_time/60.0))
                min_loss = min(min_loss, test_loss)
                # tst_summ.value.add(tag=args.data+"/Test/Loss", simple_value = test_loss)
                # tst_summ.value.add(tag=args.data+"/Test/Min_Loss", simple_value = min_loss)
                print("-------------------------------------------------------------------------------")
                # tst_writer.add_summary(tst_summ, global_iter)
                # tst_writer.flush()
        print("Best Metric: %s"%(str(max_auc) if args.task=='binary' else str(min_loss)))
        print("####################################################################################")
    print("Final Best Metric: %s"%(str(max_auc) if args.task=='binary' else str(min_loss)))
    return min_loss        

def GetEmbPred(model, fun, X, test_batch_size):
    model.eval()
    tst_len = X.shape[0]
    test_num_batch = math.ceil(tst_len / test_batch_size)
    y_preds = []
    with torch.no_grad():
        for jdx in range(test_num_batch):
            tst_st = jdx * test_batch_size
            tst_ed = min(tst_len, tst_st + test_batch_size)
            inputs = torch.from_numpy(X[tst_st:tst_ed]).to(device)
            t_preds = fun(inputs).data.cpu().numpy()
            y_preds.append(t_preds)
        y_preds = np.concatenate(y_preds, 0)
    return y_preds
