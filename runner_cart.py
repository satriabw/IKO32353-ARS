import sys
import pickle
import numpy as np
import pandas as pd

from decision_tree import RegressionTree



# Get data from train dataset
# my_data = np.genfromtxt('Train_Dataset_Mini.csv', delimiter=',', skip_header=1)



# my_data = np.genfromtxt('Train_Datasets.csv', delimiter=',',
#                         skip_header=1, dtype=None, encoding=None)

# X = np.zeros(shape=(1, 11))
# y = np.zeros(shape=(1, 1))
# for data in my_data:
#     raw = list(data)
#     data = np.array([raw[:-1]]) 
#     target = np.array([raw[-1]])
#     X = np.vstack((data, X))
#     y = np.vstack((target, y))

# X, y = my_data[:, :-1], my_data[:, -1]

# print(X)
# print(y)
models = RegressionTree()
# # X = X[:-1]
# # y = y[:-1]
models.fit(X, y)

filename = 'finalized_model.sav'
pickle.dump(models, open(filename, 'wb'))

# loaded_model = pickle.load(open(filename, 'rb'))

# Test predict for some input
# args = sys.argv[1:]
# # if len(args) < 4:
#     args = [11.65, 0.040081193, 227.0694, 2009]


# args = sys.argv[1:]
# if len(args) < 4:
#     args = ['FDT07',5.82,'reg',0,'Fruits and Vegetables',256.633,'OUT049',1999,'Medium','Tier 1','Supermarket Type1']

new_args = []
# i = 0
# for el in args:
#     if i == 0:
#         item = maps['Item_Identifier'].get(el)
#         new_args.append(item)
#     elif i == 2:
#         item = maps['Item_Fat_Content'].get(el)
#         new_args.append(item)
#     elif i == 4:
#         item = maps['Item_Type'].get(el)
#         new_args.append(item)
#     elif i == 6:
#         item = maps['Outlet_Identifier'].get(el)
#         new_args.append(item)
#     elif i == 8:
#         item = maps['Outlet_Size'].get(el)
#         new_args.append(item)
#     elif i == 9:
#         item = maps['Outlet_Location_Type'].get(el)
#         new_args.append(item)
#     elif i == 10:
#         item = maps['Outlet_Type'].get(el)
#         new_args.append(item)
#     else:
#         new_args.append(args[i])
#     i += 1

# print(new_args)

# print(loaded_model.predict(np.array([new_args])),
#       2050.664)
