import pandas as pd
from tqdm import tqdm

def clean_data(filePath):
    print('cleaning %s ...'%filePath)
    F=open(filePath+'.ssv', 'w')
    for line in tqdm(open(filePath, 'r')):
        F.write(line.replace('  ',' '))
    F.close()

root = 'data/nips/'
    
y1 = pd.read_csv(root+'AA/AA_train1.solution', header=None)
print('start clean data')
clean_data(root+'AA/AA_train1.data')
clean_data(root+'AA/AA_test1.data')
clean_data(root+'AA/AA_test2.data')
clean_data(root+'AA/AA_test3.data')
clean_data(root+'AA/AA_test4.data')
print('done')
x1 = pd.read_csv(root+'AA/AA_train1.data.ssv', sep=' ', header=None)
x2 = pd.read_csv(root+'AA/AA_test1.data.ssv', sep=' ', header=None)
x3 = pd.read_csv(root+'AA/AA_test2.data.ssv', sep=' ', header=None)
x4 = pd.read_csv(root+'AA/AA_test3.data.ssv', sep=' ', header=None)
x5 = pd.read_csv(root+'AA/AA_test4.data.ssv', sep=' ', header=None)

y1 = pd.read_csv(root+'AA/AA_train1.solution', header=None)
y2 = pd.read_csv(root+'AA/AA_test1.solution', header=None)
y3 = pd.read_csv(root+'AA/AA_test2.solution', header=None)
y4 = pd.read_csv(root+'AA/AA_test3.solution', header=None)
y5 = pd.read_csv(root+'AA/AA_test4.solution', header=None)

x1['label'] = y1
x2['label'] = y2
x3['label'] = y3
x4['label'] = y4
x5['label'] = y5

offline = pd.concat([x1,x2,x3,x4,x5])

offline.to_csv(root+'a_all.csv',index=False)

x1.to_csv(root+'a_online0.csv',index=False)
x2.to_csv(root+'a_online1.csv',index=False)
x3.to_csv(root+'a_online2.csv',index=False)
x4.to_csv(root+'a_online3.csv',index=False)
x5.to_csv(root+'a_online4.csv',index=False)
