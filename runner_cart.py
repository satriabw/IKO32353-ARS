from decision_tree import RegressionTree
import numpy as np

my_data = np.genfromtxt('Train_Datasets.csv', delimiter=',', usecols=(1, 3, 5, 7, -1), skip_header=1)
X, y = my_data[:, :-1], my_data[:, -1]

models = RegressionTree()
models.fit(X, y)