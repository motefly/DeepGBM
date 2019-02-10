for seed in 1 2 3 4 5
do
python online_main.py -data criteo_online -batch_size 4096 -plot_title 'paper_0128' \
-max_epoch 35 -nslices 20 -ntrees 200 -tree_layers 100,100,100,50 -emb_epoch 2 \
-maxleaf 128 -embsize 20 -emb_lr 1e-3 -lr 1e-3 -opt Adam -loss_de 2 -loss_dr 0.9 \
-test_batch_size 50000 -group_method Random -model gbdt -task binary
done