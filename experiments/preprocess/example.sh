python preprocess/pro_nips_A.py
python preprocess/split_train.py data/nips/a_all.csv data/nips/a_offline.csv
python preprocess/encoding_nume.py data/nipsA_offline_num/ --train_csv_path data/nips/a_offline0.csv --test_csv_path data/nips/a_offline1.csv
python preprocess/encoding_nume.py data/nipsA_online_num/ --online --data data/nips/a --num_onlines 5

python preprocess/encoding_cate.py data/nipsA_offline_cate/ --train_csv_path data/nips/a_offline0.csv --test_csv_path data/nips/a_offline1.csv
python preprocess/encoding_cate.py data/nipsA_online_cate/ --online --data data/nips/a --num_onlines 5
