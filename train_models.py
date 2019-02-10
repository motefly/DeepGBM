from models import *
from helper import *
from tqdm import tqdm
import pdb
import torch
import gc

def train_GBDT2NN(args, num_data, plot_title, key, trained_gbdt_model=None, kd_type='emb'):
    tree_layers = [int(x) for x in args.tree_layers.split(',')]
    train_x, train_y, test_x, test_y = num_data
    if trained_gbdt_model:
        gbm, train_tree_pred = trained_gbdt_model
    else:
        gbm, train_tree_pred = TrainGBDT(train_x, train_y, test_x, test_y, args.tree_lr, args.ntrees, args.maxleaf, args.mindata, args.task)
    gbms = SubGBDTLeaf_cls(train_x, test_x, gbm, args.maxleaf, num_slices=args.nslices, args = args)
    min_len_features = train_x.shape[1]
    used_features = []
    tree_outputs = []
    leaf_preds = []
    test_leaf_preds = []
    n_output = train_y.shape[1]
    max_ntree_per_split = 0
    group_average = []
    for used_feature, new_train_y, leaf_pred, test_leaf_pred, avg, all_avg in gbms:
        used_features.append(used_feature)
        min_len_features = min(min_len_features, len(used_feature))
        tree_outputs.append(new_train_y)
        leaf_preds.append(leaf_pred)
        test_leaf_preds.append(test_leaf_pred)
        group_average.append(avg)
        max_ntree_per_split = max(max_ntree_per_split, leaf_pred.shape[1])
    for i in range(len(used_features)):
        used_features[i] = sorted(used_features[i][:min_len_features])
    n_models = len(used_features)
    group_average = np.asarray(group_average).reshape(n_models, 1, 1)
    for i in range(n_models):
        if leaf_preds[i].shape[1] < max_ntree_per_split:
            leaf_preds[i] = np.concatenate([leaf_preds[i] + 1, 
                                            np.zeros([leaf_preds[i].shape[0],
                                                      max_ntree_per_split-leaf_preds[i].shape[1]],
                                                     dtype=np.int32)], axis=1)
            test_leaf_preds[i] = np.concatenate([test_leaf_preds[i] + 1, 
                                                 np.zeros([test_leaf_preds[i].shape[0],
                                                           max_ntree_per_split-test_leaf_preds[i].shape[1]],
                                                          dtype=np.int32)], axis=1)
    leaf_preds = np.concatenate(leaf_preds, axis=1)
    test_leaf_preds = np.concatenate(test_leaf_preds, axis=1)
    emb_model = EmbeddingModel(n_models, max_ntree_per_split, args.embsize, args.maxleaf+1, n_output, group_average, task=args.task).to(device)
    tree_layers.append(args.embsize)

    opt = AdamW(emb_model.parameters(), lr=args.emb_lr, weight_decay=args.l2_reg)
    tree_outputs = np.asarray(tree_outputs).reshape((n_models, leaf_preds.shape[0])).transpose((1,0))
    TrainWithLog(args, plot_title, leaf_preds, train_y, tree_outputs,
                 test_leaf_preds, test_y, emb_model, opt,
                 args.emb_epoch, args.batch_size, n_output, key+"emb-")
    output_w = emb_model.bout.weight.data.cpu().numpy().reshape(n_models*args.embsize, n_output)
    output_b = np.array(emb_model.bout.bias.data.cpu().numpy().sum())
    train_embs = GetEmbPred(emb_model, emb_model.lastlayer, leaf_preds, args.test_batch_size)
    del tree_outputs, leaf_preds, test_leaf_preds
    gc.collect()
    concate_train_x = np.concatenate([train_x, np.zeros((train_x.shape[0],1), dtype=np.float32)], axis=-1)
    concate_test_x = np.concatenate([test_x, np.zeros((test_x.shape[0],1), dtype=np.float32)], axis=-1)
    tree_outputs = train_embs
    for seed in args.seeds:
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        gbdt2nn_model = GBDT2NN(concate_train_x.shape[1], 
                                np.asarray(used_features,dtype=np.int64),
                                tree_layers,
                                output_w, output_b, args.task).to(device)
        opt = AdamW(gbdt2nn_model.parameters(), lr=args.lr, weight_decay=args.l2_reg, amsgrad=False)

        TrainWithLog(args, plot_title+'seed'+str(seed), concate_train_x, train_y, tree_outputs,
                        concate_test_x, test_y, gbdt2nn_model, opt,
                        args.max_epoch, args.batch_size, n_output, key)
        _,pred_y = EvalTestset(concate_test_x, test_y, gbdt2nn_model, args.test_batch_size)
        metric = eval_metrics(args.task, test_y, pred_y)
        print('Final metrics: %s'%str(metric))
    return gbdt2nn_model, opt, metric


def train_cateModels(args, cate_data, plot_title, key):
    train, test = cate_data
    train_x, train_y, feature_sizes = train
    test_x, test_y, _ = test
    # feature_size = max(feature_sizes)
    field_size = train_x.shape[1]
    cate_layers = [int(x) for x in args.cate_layers.split(',')]
    for seed in args.seeds:
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        if args.model == 'deepfm':
            model = DeepFM(field_size,feature_sizes,deep_layers=cate_layers,
                            use_cuda=True,use_fm=True,use_deep=True,
                            embedding_size=args.cate_embsize,task=args.task).cuda()
        elif args.model == 'wideNdeep':
            model = DeepFM(field_size,feature_sizes,deep_layers=cate_layers,
                            use_cuda=True,use_fm=False,use_deep=True,use_wide=True,
                            embedding_size=args.cate_embsize,task=args.task).cuda()
        elif args.model == 'lr':
            model = DeepFM(field_size,feature_sizes,deep_layers=cate_layers,
                            use_cuda=True,use_fm=False,use_deep=False,use_wide=True,
                            embedding_size=args.cate_embsize,task=args.task).cuda()
        elif args.model == 'fm':
            model = DeepFM(field_size,feature_sizes,deep_layers=cate_layers,
                            use_cuda=True,use_fm=True,use_deep=False,use_wide=False,
                            embedding_size=args.cate_embsize,task=args.task).cuda()
        elif args.model == 'pnn':
            model = PNN(field_size, feature_sizes,deep_layers=cate_layers,
                            use_cuda=True, task=args.task,
                            use_inner_product=True, use_outer_product=True).cuda()

        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg, amsgrad=False)
        TrainWithLog(args, plot_title+'seed'+str(seed), train_x, train_y, None,
                        test_x, test_y, model,
                        opt, args.max_epoch, args.batch_size, 1, key)

        _,pred_y = EvalTestset(test_x, test_y, model, args.test_batch_size)
        metric = eval_metrics(args.task, test_y, pred_y)
        print('Final metrics: %s'%str(metric))
    return model, opt, metric

def train_DEEPGBM(args, num_data, cate_data, plot_title, key, trained_gbdt_model=None):
    tree_layers = [int(x) for x in args.tree_layers.split(',')]
    cate_layers = [int(x) for x in args.cate_layers.split(',')]
    train_x, train_y, test_x, test_y = num_data
    if trained_gbdt_model:
        gbm, train_tree_pred = trained_gbdt_model
    else:
        gbm, train_tree_pred = TrainGBDT(train_x, train_y, test_x, test_y, args.tree_lr, args.ntrees, args.maxleaf, args.mindata, args.task)
    gbms = SubGBDTLeaf_cls(train_x, test_x, gbm, args.maxleaf, num_slices=args.nslices, args = args)
    min_len_features = train_x.shape[1]
    used_features = []
    tree_outputs = []
    leaf_preds = []
    test_leaf_preds = []
    n_output = train_y.shape[1]
    max_ntree_per_split = 0
    group_average = []
    for used_feature, new_train_y, leaf_pred, test_leaf_pred, avg, all_avg in gbms:
        used_features.append(used_feature)
        min_len_features = min(min_len_features, len(used_feature))
        tree_outputs.append(new_train_y)
        leaf_preds.append(leaf_pred)
        test_leaf_preds.append(test_leaf_pred)
        group_average.append(avg)
        max_ntree_per_split = max(max_ntree_per_split, leaf_pred.shape[1])
    for i in range(len(used_features)):
        used_features[i] = sorted(used_features[i][:min_len_features])
    n_models = len(used_features)
    group_average = np.asarray(group_average).reshape(n_models, 1, 1)

    for i in range(n_models):
        if leaf_preds[i].shape[1] < max_ntree_per_split:
            leaf_preds[i] = np.concatenate([leaf_preds[i] + 1, 
                                            np.zeros([leaf_preds[i].shape[0],
                                            max_ntree_per_split-leaf_preds[i].shape[1]],
                                            dtype=np.int32)], axis=1)
            test_leaf_preds[i] = np.concatenate([test_leaf_preds[i] + 1, 
                                                 np.zeros([test_leaf_preds[i].shape[0],
                                                max_ntree_per_split-test_leaf_preds[i].shape[1]],
                                                dtype=np.int32)], axis=1)
    leaf_preds = np.concatenate(leaf_preds, axis=1)
    test_leaf_preds = np.concatenate(test_leaf_preds, axis=1)
    emb_model = EmbeddingModel(n_models, max_ntree_per_split, args.embsize, args.maxleaf+1, n_output, group_average, task=args.task).to(device)
    tree_layers.append(args.embsize)
    opt = AdamW(emb_model.parameters(), lr=args.emb_lr, weight_decay=args.l2_reg)
    tree_outputs = np.asarray(tree_outputs).reshape((n_models, leaf_preds.shape[0])).transpose((1,0))
    TrainWithLog(args, plot_title, leaf_preds, train_y, tree_outputs,
                 test_leaf_preds, test_y, emb_model, opt,
                 args.emb_epoch, args.batch_size, n_output, key+"emb-")
    output_w = emb_model.bout.weight.data.cpu().numpy().reshape(n_models*args.embsize, n_output)
    output_b = np.array(emb_model.bout.bias.data.cpu().numpy().sum())
    train_embs = GetEmbPred(emb_model, emb_model.lastlayer, leaf_preds, args.test_batch_size)
    del tree_outputs, leaf_preds, test_leaf_preds
    gc.collect()
    tree_outputs = train_embs
    # cate_model dataset loading
    train, test = cate_data
    train_xc, _, feature_sizes = train
    test_xc, _, _ = test
    field_size = train_xc.shape[1]
    concate_train_x = np.concatenate([train_x, np.zeros((train_x.shape[0],1), dtype=np.float32)], axis=-1)
    concate_test_x = np.concatenate([test_x, np.zeros((test_x.shape[0],1), dtype=np.float32)], axis=-1)
    del train_x, test_x
    gc.collect()
    for seed in args.seeds:
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        deepgbm_model = DeepGBM(concate_train_x.shape[1],np.asarray(used_features,dtype=np.int64),
                                tree_layers, output_w, output_b, args.task,
                                field_size, feature_sizes,
                                embedding_size=args.embsize).to(device)
        opt = AdamW(deepgbm_model.parameters(), lr=args.lr, weight_decay=args.l2_reg, amsgrad=False, model_decay_opt=deepgbm_model, weight_decay_opt=args.l2_reg_opt, key_opt='deepfm')
        TrainWithLog(args, plot_title+'seed'+str(seed), concate_train_x, train_y, tree_outputs,
                     concate_test_x, test_y, deepgbm_model, opt,
                     args.max_epoch, args.batch_size, n_output, key,
                     train_x_opt=train_xc, test_x_opt=test_xc)
        _,pred_y = EvalTestset(concate_test_x, test_y, deepgbm_model, args.test_batch_size, test_x_opt=test_xc)
        metric = eval_metrics(args.task, test_y, pred_y)
        print('Final metrics: %s'%str(metric))
    return deepgbm_model, opt, metric

def train_D1(args, num_data, cate_data, plot_title, key="", trained_gbdt_model=None):
    train_x, train_y, test_x, test_y = num_data
    cate_layers = [int(x) for x in args.cate_layers.split(',')]
    if trained_gbdt_model:
        gbm, trn_pred = trained_gbdt_model
    else:
        gbm, trn_pred = TrainGBDT(train_x, train_y, test_x, test_y, args.tree_lr, args.ntrees, args.maxleaf, args.mindata, args.task)
    tst_pred = gbm.predict(test_x, raw_score=True)
    del train_x, test_x
    gc.collect()
    train, test = cate_data
    train_xc, _, feature_sizes = train
    test_xc, _, _ = test
    field_size = train_xc.shape[1]
    for seed in args.seeds:
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        deepgbm_model = DeepGBM(task=args.task,
                                cate_field_size=field_size, feature_sizes=feature_sizes,
                                embedding_size=args.embsize, num_model='gbdt',
                                maxleaf=args.maxleaf,deep_layers=cate_layers).to(device)
        opt = AdamW(deepgbm_model.parameters(), lr=args.lr, weight_decay=args.l2_reg, amsgrad=False, model_decay_opt=deepgbm_model, weight_decay_opt=args.l2_reg_opt, key_opt='deepfm')
        TrainWithLog(args, plot_title+'seed'+str(seed), trn_pred, train_y, None,
                     tst_pred, test_y, deepgbm_model, opt,
                     args.max_epoch, args.batch_size, 1, key,
                     train_x_opt=train_xc, test_x_opt=test_xc)
    _,pred_y = EvalTestset(tst_pred, test_y, deepgbm_model, args.test_batch_size, test_x_opt=test_xc)
    metric = eval_metrics(args.task, test_y, pred_y)
    print('Final metrics: %s'%str(metric))
    return deepgbm_model, opt, metric, gbm
