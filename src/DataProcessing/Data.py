import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, load_wine
import numpy as np

from src.DataProcessing.Data_Transformation import Data_Transformation


class Data:
    """
    Add Description
    """

    def __init__(self, data_params, sep=','):
        self.features_dict = {}
        self.dataset = data_params['dataset']
        # self.data_path = data_params['path'] + data_params['filename']
        self.sep = sep
        self.ID = None
        self.X_train = None
        self.y_train = None
        self.X_cv = None
        self.y_cv = None
        self.X_test = None
        self.y_test = None
        self.features = None
        self.transformation_method = data_params['normalization']

    def process(self, root_dir):
        """
        Add description here
        """
        if self.dataset == 'iris':
            data, target = load_iris(return_X_y=True, as_frame=True)
        elif self.dataset == 'digits':
            data, target = load_digits(return_X_y=True, as_frame=True)
        elif self.dataset == 'breast_cancer':
            data, target = load_breast_cancer(return_X_y=True, as_frame=True)
        elif self.dataset == 'wine':
            data, target = load_wine(return_X_y=True, as_frame=True)
        else:
            print('Wrong dataset input')
            return
        # Transform data
        transformer = Data_Transformation(self.transformation_method)
        transformed_data = transformer.transform_data(data)

        # WARNING: TEST has more data than CV
        # split test CV
        X_train, self.X_test, y_train, self.y_test = train_test_split(transformed_data, target, test_size=0.2)
        self.X_train, self.X_cv, self.y_train, self.y_cv = train_test_split(X_train, y_train, test_size=0.2)

        self.features = self.X_train.columns
        for i in range(len(self.features)):
            self.features_dict[i] = self.features[i]

