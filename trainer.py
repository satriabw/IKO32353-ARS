import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from decision_tree import RegressionTree
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor

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
my_data = pd.read_csv('Test_Dataset_Cleansed.csv')
features = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

transformer = DatasetTransformer(my_data, features)
my_data, maps = transformer.transform()

X = my_data.values

# models = DecisionTreeRegressor()
# models.fit(X_train, y_train)


filename = 'finalized_model.sav'
# # pickle.dump(models, open(filename, 'wb'))
models = pickle.load(open(filename, 'rb'))
y_pred = models.predict(X)

my_data["Item_Outlet_Sales"] = y_pred
my_data.to_csv('jawaban.csv')