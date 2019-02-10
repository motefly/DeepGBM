for seed in 1 2 3 4 5
do
python online_main.py -data nipsA_online -batch_size 512 -plot_title 'paper_0201f' \
-max_epoch 20 -lr 1e-3 -opt Adam -test_batch_size 5000 -model deepfm \
-task binary -l2_reg 1e-6 -test_freq 3000 -seed $seed\
-cate_layers 16,16,16 -nslices 5 \
-cate_embsize 2 -log_freq 500 -online_bz 512 -online_epoch 1
done