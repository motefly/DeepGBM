for seed in 1 2 3 4 5
do
python online_main.py -data flight_online -batch_size 512 \
-plot_title 'paper_0131_f30' -max_epoch 45 -nslices 20 -ntrees 200 \
-tree_layers 100,100,100,50 -emb_epoch 2 -maxleaf 128 -embsize 20 \
-emb_lr 1e-3 -lr 1e-3 -opt Adam -loss_de 2 -loss_dr 0.7 -test_batch_size 5000 \
-group_method Random -model deepgbm -feat_per_group 128  \
-task binary -tree_lr 0.15 -l2_reg 1e-6 -test_freq 3000 -seed $seed-cate_layers 32,32 \
       -online_bz 512 -online_epoch 1

python online_main.py -data flight_online -batch_size 512 \
-plot_title 'paper_0131_f30' -max_epoch 45 -nslices 20 -ntrees 200 \
-tree_layers 100,100,100,50 -emb_epoch 2 -maxleaf 128 -embsize 20 \
-emb_lr 1e-3 -lr 1e-3 -opt Adam -loss_de 2 -loss_dr 0.7 -test_batch_size 5000 \
-group_method Random -model deepgbm -feat_per_group 128  \
-task binary -tree_lr 0.15 -l2_reg 1e-6 -test_freq 3000 -seed $seed-cate_layers 32,32 \
-online_bz 512 -online_epoch 1 -offline
done