for seed in 1 2 3 4 5
do
python online_main.py -data flight_online -batch_size 512 -plot_title 'paper_0201' \
-max_epoch 12 -lr 1e-3 -opt Adam -test_batch_size 5000 -model wideNdeep \
-task binary -l2_reg 1e-6 -seed $seed-test_freq 3000 -cate_layers 32,32 \
-online_bz 512 -online_epoch 1
done
