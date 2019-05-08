for seed in 1 2 3 4 5
do
python online_main.py -data criteo_online -batch_size 4096 -plot_title 'paper_0130' \
-max_epoch 18 -lr 1e-3 -opt Adam -test_batch_size 50000 -model pnn \
-task binary -l2_reg 1e-6 -seed $seed-test_freq 3000 -cate_layers 32,32,32 -online_bz 4096 \
-online_epoch 1
done