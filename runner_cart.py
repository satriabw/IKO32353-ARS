import sys

import numpy as np

from decision_tree import RegressionTree

# Get data from train dataset
my_data = np.genfromtxt('Train_Dataset_Mini.csv', delimiter=',',
                        skip_header=1, dtype=None, encoding=None)
X, y = my_data[:, :-1], my_data[:, -1]

models = RegressionTree()
models.fit(X, y)
# Test predict for some input
args = sys.argv[1:]
if len(args) < 4:
    args = [11.65, 0.040081193, 227.0694, 2009]

print(models.predict(np.array([args])),
      1141.847)
