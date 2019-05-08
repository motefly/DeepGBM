python main.py -data flight_offline -batch_size 512 -plot_title 'paper_0201' -max_epoch 45 \
-nslices 20 -ntrees 200 -maxleaf 128 -lr 1e-3 -opt Adam -test_batch_size 5000 -model d1 \
-task binary -tree_lr 0.15 -l2_reg 1e-6
