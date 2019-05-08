python main.py -data nipsA_offline -batch_size 512 -plot_title 'paper_0201_f' \
-max_epoch 20 -lr 1e-3 -opt Adam -test_batch_size 5000 -model wideNdeep \
-task binary -l2_reg 1e-6 -test_freq 3000 -seed 1,2,3,4,5 \
-cate_layers 16,16 -nslices 5 \
-cate_embsize 2 -log_freq 500
