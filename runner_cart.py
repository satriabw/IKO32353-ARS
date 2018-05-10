import sys
import pickle
import numpy as np
import pandas as pd

from decision_tree import RegressionTree

# Get data from train dataset
my_data = np.genfromtxt('./datasets/Train_Datasets.csv', delimiter=',', skip_header=1, dtype=None, encoding=None)
X = np.zeros(shape=(1, 11))
y = np.zeros(shape=(1, 1))
for data in my_data:
    raw = list(data)
    data = np.array([raw[:-1]]) 
    target = np.array([raw[-1]])
    X = np.vstack((data, X))
    y = np.vstack((target, y))

X, y = my_data[:, :-1], my_data[:, -1]
models = RegressionTree()
# X = X[:-1]
# y = y[:-1]
models.fit(X, y)

filename = 'models_version_one.sav'
pickle.dump(models, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

# Test predict for some input
args = sys.argv[1:]
if len(args) < 4:
    args = ['FDT07',5.82,'reg',0,'Fruits and Vegetables',256.633,'OUT049',1999,'Medium','Tier 1','Supermarket Type1']
print(loaded_model.predict(np.array([args])),
      2050.664)
