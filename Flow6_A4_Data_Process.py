# Split files : A4 Benchmark
import pandas as pd
from os import listdir
from os.path import isfile, join

path = 'C:/Users/Roshan/PycharmProjects/Yahoo/Webscope/A4Benchmark/'
csvfiles = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]

train = pd.DataFrame()
test = pd.DataFrame()

# for file in csvfiles[:-20]:
for file in csvfiles[20:]:
    train = train.append(pd.read_csv(path + file))

# for file in csvfiles[-20:]:
for file in csvfiles[:20]:
    test = test.append(pd.read_csv(path + file))

del train['timestamps']
del test['timestamps']

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
