for seed in 1 2 3 4 5
do
python online_main.py -data flight_online -batch_size 512 -plot_title 'paper_0131_f3' -max_epoch 45 -nslices 20 -ntrees 200 -maxleaf 128 -lr 1e-3 -opt Adam -test_batch_size 5000 -model d1 -task binary -tree_lr 0.15 -l2_reg 1e-6 -online_bz 512 -online_epoch 1 -seed 1
done