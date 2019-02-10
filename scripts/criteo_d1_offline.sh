python main.py -data criteo_offline -batch_size 4096 -plot_title 'paper_0129' \
-max_epoch 20 -nslices 20 -ntrees 200 -maxleaf 128 -lr 1e-3 -opt Adam -test_batch_size 5000 \
-model d1 -task binary -tree_lr 0.15 -l2_reg 1e-6 -cate_layers 32,32 -seed 1,2,3,4,5
