#for seed in 1 2 3 4 5
#do
    python main.py -data criteo_offline -batch_size 4096 -plot_title 'paper_0127' -max_epoch 30 \
     -nslices 20 -ntrees 200 -tree_layers 100,100,100,50 -emb_epoch 2 -maxleaf 128 -embsize 20 -emb_lr 1e-3 \
     -lr 1e-3 -opt Adam -loss_de 3 -loss_dr 0.9 -loss_init 0.9 -test_batch_size 50000 -group_method Random -model deepgbm \
     -feat_per_group 128  -task binary -tree_lr 0.15 -l2_reg 1e-6 -test_freq 3000 \
     -cate_layers 32,32 -seed 1,2,3,4,5
#done
