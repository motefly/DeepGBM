for seed in 1 2 3 4 5
do
python online_main.py -data flight_online -batch_size 512 -plot_title 'paper_0129_f33' \
-max_epoch 45 -lr 1e-3 -opt Adam -test_batch_size 5000 -model deepfm \
-task binary -l2_reg 1e-6 -seed $seed-test_freq 3000 -online_epoch 1 -online_bz 512 -cate_layers 32,32
done