import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from decision_tree import RegressionTree
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

class DatasetTransformer():
    def __init__(self, dataset, features):
        self.dataset = dataset
        self.features = features

    def _mapGenerator(self, dataset, feature):
        mapper = {}
        category = dataset[feature].unique()
        for i, item in enumerate(category):
            mapper[item] = i
        return mapper

    def _transformCategorical(self, dataset, features):
        newDataset = []
        mappers = {}
        for feature in features:
            if dataset[feature].dtype != 'object':
                continue
            mappers.update({feature: self._mapGenerator(dataset, feature)})

        newDataset = dataset.replace(mappers)
        return newDataset, mappers
    
    def transform_to_csv(self):
        dataset, maps = self._transformCategorical(self.dataset, self.features)
        dataset.to_csv('Train_Dataset_Transformed.csv')

    def transform(self):
        dataset, maps = self._transformCategorical(self.dataset, self.features)
        return dataset, maps

# Get data from train dataset
my_data = pd.read_csv('./datasets/Datasets_Cleaned.csv')
msk = np.random.rand(len(my_data)) < 0.8

size = my_data.shape[1]

train = my_data[msk].values
test = my_data[~msk].values
X_train, y_train = train[:, 1:-1], train[:, -1]
X_test, y_test = test[:, 1:-1], test[:, -1]


models_dtreg = DecisionTreeRegressor()
models_dtreg.fit(X_train, y_train)

models_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
models_svr.fit(X_train, y_train)

filename = 'models_finals.sav'
# pickle.dump(models, open(filename, 'wb'))
models_cart = pickle.load(open(filename, 'rb'))

y_pred_cart = models_cart.predict(X_test)
y_pred_dtreg = models_dtreg.predict(X_test)
y_pred_svr = models_svr.predict(X_test)

print("Decision Tree Regressor: ", np.sqrt(metrics.mean_squared_error(y_pred_dtreg, y_test)))
print("Support Vector Machine Regressor: ", np.sqrt(metrics.mean_squared_error(y_pred_svr, y_test)))
print("Our Models: ", np.sqrt(metrics.mean_squared_error(y_pred_cart, y_test)))




# new_args = []
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