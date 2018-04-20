from decision_tree import RegressionTree
import numpy as np

# Get data from train dataset
my_data = np.genfromtxt('Train_Dataset_Mini.csv', delimiter=',', usecols=(1, 3, 5, 7, -1), skip_header=1)
X, y = my_data[:, :-1], my_data[:, -1]

models = RegressionTree()
models.fit(X, y)
# Test predict for some input
print(models.predict(np.array([[11.65, 0.040081193, 227.0694, 2009]])), 1141.847)
