import sys

import numpy as np
import pandas as pd

from decision_tree import RegressionTree

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
# my_data = np.genfromtxt('Train_Dataset_Mini.csv', delimiter=',', skip_header=1)
my_data = pd.read_csv('Train_Dataset_Mini.csv')
features = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

transformer = DatasetTransformer(my_data, features)
transformer.transform_to_csv()



# X, y = my_data[:, :-1], my_data[:, -1]

# models = RegressionTree()
# models.fit(X, y)
# # Test predict for some input
# args = sys.argv[1:]
# if len(args) < 4:
#     args = [11.65, 0.040081193, 227.0694, 2009]

# print(models.predict(np.array([args])),
#       1141.847)
