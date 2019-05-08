# -*- coding:utf-8 -*-

"""
Created on Dec 10, 2017
@author: jachin,Nie

This script is used to sample data from the raw dataset

python sample.py s_path t_path prob

"""

import argparse
import sys
import random
from tqdm import tqdm
import random
random.seed(1)
s_path = sys.argv[1]
# tr_path = sys.argv[2]
te_path = sys.argv[2]
# prob = float(sys.argv[4])

# with open(tr_path,'wb') as fr:
#     with open(te_path,'wb') as fe:
#         for line in open(s_path,'rb'):
#             if random.random() < prob:
#                 fr.write(line)
#             else:
#                 fe.write(line)
fes = []
thres = []
with open(s_path, 'rb') as raw:
    lines = raw.readlines()
rates = [0.9, 0.1]
#rates = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
Sum = 0
for idx, rate in enumerate(rates):
    fes.append(open(te_path.split('.')[0] + str(idx) + '.' + te_path.split('.')[-1], 'wb'))
    Sum += rate
    thres.append(Sum * len(lines))

idx = 0
for cnt,line in enumerate(tqdm(lines)):
    if cnt == 0:
        for f in fes:
            f.write(line)
        continue
    # cnt = random.random()
    # if cnt > 0.9:
    #     fes[1].write(line)
    # else:
    #     fes[0].write(line)
    if cnt > thres[idx]:
        idx += 1
    fes[idx].write(line)
    
for idx, rate in enumerate(rates):
    fes[idx].close()
