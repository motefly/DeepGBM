# DeepGBM

Implementation for the paper "DeepGBM: A Deep Learning Framework Distilled  by GBDT for Online Prediction Tasks", 
which has been accepted by KDD'2019 as an Oral Paper, in the Research Track.

If you find this code useful in your research, please cite the paper (to appear):

Guolin Ke, Zhenhui Xu, Jia Zhang, Jiang Bian, and Tie-yan Liu. "DeepGBM: A Deep Learning Framework Distilled  by GBDT for Online Prediction Tasks." In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, ACM, 2019.

## Introduction
This repo is built for the experimental codes in our paper, 
containing all the data preprocessing, baseline models implementation
and proposed model implementation.
For GBDT based model, our implementation is based on LightGBM. 
For the NN based model, our implementation is based on pytorch.

There are five main folders in the project, `data` is for data storage, 
`models` contains all the implementation codes of the proposed model 
and other baseline models, `preprocess` is the folder containing
feature selection and encoding, `online_scripts`
(online experiments) and `scripts` (offline experiments) stores the
scripts/parameters of our experiments. 

Moreover, `main.py` is the entry code file for offline 
experiments while `online_main.py` is the entry file for online one.
Besides, `data_helpers.py` contains the data loader, `helper.py`
contains the general training and testing logic for NN. 
`train_models.py` is for the specific training process of each model.
In `models`, there are 5 main models, which are *DeepGBM* (*DeepGBM* and its *D1* implementation), 
*GBDT2NN*, *PNN*, *DeepFM* (*DeepFM* and *Wide&Deep*) and *Components* (*GBDT*
and some basic structures). `tree_model_interpreter.py` is used for interpreting
the trained GBDT's structure.

## Environment Setting
The main dependency is shown as follows:
* Python==3.6.6
* LightGBM==2.2.1
* Pytorch==0.4.1
* Sklearn==0.19.2

## Data preprocess
We benchmark 7 datasets in our paper, containing 
[Flight](http://stat-computing.org/dataexpo/2009/),
[Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge/data),
[Malware](https://www.kaggle.com/c/malware-classification),
[Nips-A,B,D](https://www.4paradigm.com/competition/nips2018) (AutoML-1,2,3) and
[Zillow](https://www.kaggle.com/c/zillow-prize-1). You can download
these datasets from their links and preprocess them consulting `preprocess/example.sh`.
```bash
# Please first download the AutoML Dataset and put it in data/nips (make data/nips/AA exists or change the root in the code file)
python preprocess/pro_nips_A.py
# to split the original set to 2 offline set (one for offline training and the other for offline testing)
python preprocess/split_train.py data/nips/a_all.csv data/nips/a_offline.csv
# encode all the features to numerical features
python preprocess/encoding_nume.py data/nipsA_offline_num/ --train_csv_path data/nips/a_offline0.csv --test_csv_path data/nips/a_offline1.csv
python preprocess/encoding_nume.py data/nipsA_online_num/ --online --data data/nips/a --num_onlines 5
# encode all the features to categorical features
python preprocess/encoding_cate.py data/nipsA_offline_cate/ --train_csv_path data/nips/a_offline0.csv --test_csv_path data/nips/a_offline1.csv
python preprocess/encoding_cate.py data/nipsA_online_cate/ --online --data data/nips/a --num_onlines 5
```

All the datasets should be converted into *.csv* files (like `preprocess/pro_nips_A.py`) first and then processed 
by encoders in `preprocess`. The features used for each dataset could be seen in 
`preprocess/encoding_*.py`, the main function specifically. The other file,
`split_train.py` is used for general dataset splitting, excepts *Flight*
in offline phase, *Flight* and *AutoML-1* in online phase, whose splitting
has been introduced in the paper.

## Offline and online evaluation
We evaluate all the models on all the above datasets 
in offline phase and 3 datasets in online one. Consulting the
parameter setting in `scripts`, you could reproduce our experimental
results in the paper. For example,
```bash
python main.py -data nipsA_offline -batch_size 512 -plot_title '0201' \
-max_epoch 20 -lr 1e-3 -opt Adam -test_batch_size 5000 -model gbdt2nn \
-task binary -l2_reg 1e-6 -test_freq 3000 -seed 1,2,3,4,5 -group_method Random \
-emb_epoch 2 -loss_de 2 -loss_dr 0.7 -tree_lr 0.1 -cate_layers 16,16 -nslices 5 \
-cate_embsize 2 -tree_layers 100,100,100,50 -embsize 20 -maxleaf 64 -log_freq 500
```
We can easily get part of the results on nipsA (AutoML-1) as the following table.

| Model | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean | Std |
|--|--|--|--|--|--|--|--|
| DeepGBM | 0.7564 | 0.7564 | 0.7566 | 0.7564 | 0.7562 | 0.7564 | 1e-4|
| DeepGBM (D1) | 0.7538 | 0.7538 | 0.7539 | 0.7541 | 0.7536 | 0.7538 | 2e-4 | 
| DeepGBM (D2) | 0.7557 | 0.7557 | 0.7560 | 0.7557 | 0.7556 | 0.7557 | 2e-4 |

Note that DeepGBM (D2) is named GBDT2NN in our codes.

Similarly, you can also run the scripts in 
`online_scripts` for online evalutation. 
```bash
python online_main.py -data criteo_online -batch_size 4096 -plot_title '0201' -max_epoch 12 \
-nslices 20 -ntrees 200 -tree_layers 100,100,100,50 -emb_epoch 2 -maxleaf 128 -embsize 20 -emb_lr 1e-3 \
-lr 1e-3 -opt Adam -loss_de 2 -loss_dr 0.9 -test_batch_size 50000 -group_method Random -model deepgbm \
-feat_per_group 128 -task binary -tree_lr 0.15 -l2_reg 1e-6 -test_freq 3000 \
-cate_layers 32,32 -seed 1 -online_bz 4096 -online_epoch 1
```

Thanks for your visiting, and if you have any questions, please new an issue.