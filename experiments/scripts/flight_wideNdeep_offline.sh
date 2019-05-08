# for seed in 1 2 3 4 5
# do
    python main.py -data flight_offline -batch_size 512 -plot_title 'paper_0124' -max_epoch 45 -lr 1e-3 -opt Adam -test_batch_size 5000 -model wideNdeep -task binary -l2_reg 1e-6 -seed 1,2,3,4,5 -test_freq 3000
# done
