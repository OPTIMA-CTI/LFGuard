import pandas as pd
import glob as glob

import os
import re #for patter matching
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import joblib

# files = sorted(glob.glob('models/300*.parquet'))
files = sorted(glob.glob('models/300*.par'))
print(len(files))
# data_all = pd.concat((pd.read_parquet(file).iloc[:, :-1].assign(filename=file) for file in files), ignore_index=True).iloc[:, :-1]
data_all = pd.concat((pd.read_parquet(file).iloc[:, :-1].assign(filename=file, targets=lambda x: x['targets'].apply(lambda y: int(y[0]))) for file in files), ignore_index=True).iloc[:, :-1]

print(data_all.describe())

x = data_all.drop(['targets'],axis=1)
print(x.head())
y = data_all['targets']
print(y.head())



classes=y.unique()
clf = SGDClassifier(loss='hinge', max_iter=1000)
batch_size = 32
n_batches = len(x) // batch_size + 1
print('no.of batchs=',n_batches)

X_batch = x.iloc[:batch_size,:]
y_batch = y.iloc[:batch_size]
clf.fit(X_batch,y_batch)
print('initialized fit')

for i in range(1,n_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    X_batch = x.iloc[start_idx:end_idx, :]
    y_batch = y.iloc[start_idx:end_idx]
    if i%50 ==0:
      print('training at round',i)
    kk = len(X_batch)
    print(kk)
    if (kk==0):
      break
    clf.partial_fit(X_batch, y_batch, classes = classes)


joblib.dump(clf, 'gtsr_model_f_r50_le1.pkl')







