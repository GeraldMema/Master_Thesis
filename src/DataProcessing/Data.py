import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from src.DataProcessing.Data_Transformation import Data_Transformation

class Data:
    """
    Add Description
    """

    def __init__(self, data_params, sep=','):
        self.data_path = data_params['path'] + data_params['filename']
        self.sep = sep
        self.target = data_params['target']
        self.ID = None
        self.Id_name = data_params['ID']
        self.X_train = None
        self.y_train = None
        self.X_cv = None
        self.y_cv = None
        self.X_test = None
        self.y_test = None
        self.transformation_method = data_params['normalization']

    def process(self):
        # read data
        data = pd.read_csv(self.data_path, self.sep)

        # delete the primary key
        self.ID = data[self.Id_name]
        del data[self.Id_name]

        # label encoding
        le = preprocessing.LabelEncoder()
        data[self.target] = le.fit_transform(data[self.target])

        # Transform data
        transformer = Data_Transformation('MinMax')
        transformer.transform_data(data, self.target)

        # WARNING: TEST has more data than CV
        # split test CV
        train, test_data = train_test_split(transformer.transformed_data, test_size=0.2)
        train_data, cv_data = train_test_split(train, test_size=0.2)

        self.y_train = train_data[self.target]
        self.X_train = train_data.drop([self.target], axis=1)
        self.y_test = test_data[self.target]
        self.X_test = test_data.drop([self.target], axis=1)
        self.y_cv = cv_data[self.target]
        self.X_cv = cv_data.drop([self.target], axis=1)
