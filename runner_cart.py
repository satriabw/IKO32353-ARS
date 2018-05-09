import sys

import numpy as np

from decision_tree import RegressionTree

# Get data from train dataset
my_data = np.genfromtxt('Train_Dataset_Mini.csv', delimiter=',',
                        skip_header=1, dtype=None, encoding=None)

X = np.zeros(shape=(1, 11))
y = np.zeros(shape=(1, 1))
for data in my_data:
    raw = list(data)
    data = np.array([raw[:-1]]) 
    target = np.array([raw[-1]])
    X = np.vstack((data, X))
    y = np.vstack((target, y))

# X, y = my_data[:, :-1], my_data[:, -1]

models = RegressionTree()
X = X[:-1]
y = y[:-1]
models.fit(X, y)
# Test predict for some input
args = sys.argv[1:]
if len(args) < 4:
    args = ['FDV11',9.1,'Regular',0,'Breads',173.2054,'OUT045',2002,'Medium','Tier 2','Supermarket Type1']

print(models.predict(np.array([args])),
      1141.847)
