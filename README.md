# DeepGBM

Implementation for the paper "DeepGBM: A Deep Learning Framework Distilled  by GBDT for Online Prediction Tasks", 
which has been accepted by KDD'2019 as an Oral Paper, in the Research Track. You can get more information from the [video](https://www.youtube.com/watch?v=UzXNzW2s8Pw) and the [paper](https://www.kdd.org/kdd2019/accepted-papers/view/deepgbm-a-deep-learning-framework-distilled-by-gbdt-for-online-prediction-t).

If you find this code useful in your research, please cite the [paper](https://www.kdd.org/kdd2019/accepted-papers/view/deepgbm-a-deep-learning-framework-distilled-by-gbdt-for-online-prediction-t):

Guolin Ke, Zhenhui Xu, Jia Zhang, Jiang Bian, and Tie-Yan Liu. "DeepGBM: A Deep Learning Framework Distilled by GBDT for Online Prediction Tasks." In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2019: 384-394.

## Brief Introduction
This repo is built for the experimental codes in our paper, 
containing all the data preprocessing, baseline models implementation
and proposed model implementation ([full codes here](https://github.com/motefly/DeepGBM/tree/master/experiments)). For quick start, here we only show the codes related to our model.
For GBDT based model, our implementation is based on LightGBM. 
For NN based model, our implementation is based on pytorch.

There are three main folders in the project, `data` is for data storage, `preprocess` is the folder containing feature selection and encoding, `models` contains all the implementation codes of the proposed model.
For more detailed experiments codes, refer to the [`experiments`](https://github.com/motefly/DeepGBM/tree/master/experiments) folder. 

Besides, `main.py` is the entry code file for our model.
Besides, `data_helpers.py` contains the data loader, `helper.py`
contains the general training and testing logic for NN.
`train_models.py` is for the specific training process of the model.
In `models`, there are some implementations of main models. `tree_model_interpreter.py` is used for interpreting
the trained GBDT's structure.

## Environment Setting
The main dependency is shown as follows:
* Python==3.6.6
* LightGBM==2.2.1
* Pytorch==0.4.1
* Sklearn==0.19.2

## Quick Start
All the datasets should be converted into *.csv* files first and then processed by encoders in `preprocess`. The features used for each dataset could be seen in `preprocess/encoding_*.py`, the main function specifically.

To run DeepGBM, after the above step, you will prepare your data in *.npy* format. Then we can use the function in `data_helpers.py` to load its numerical part and categorical part:
```python
num_data = dh.load_data(args.data+'_num')
cate_data = dh.load_data(args.data+'_cate')
# following is designed for faster catNN inputs
cate_data = dh.trans_cate_data(cate_data)
```
On the contrary, if you run GBDT2NN or CatNN only, you can only feed the numerical data or categorical data into the model.
Then, you can call the functions in `train_models.py` like:
```python
train_GBDT2NN(args, num_data, plot_title)
# or
train_DEEPGBM(args, num_data, cate_data, plot_title)
```

Thanks for your visiting, and if you have any questions, please new an issue.
