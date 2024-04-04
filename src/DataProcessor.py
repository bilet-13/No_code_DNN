import pandas as pd
import torch
from sklearn.model_selection import train_test_split

TEST_SIZE_PERCENT = 0.3
RANDOM_SEED = 42

class DataProcessor():
    def __init__(self):
        self.data = None
        self.input_columns = None
        self.target_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, file_name, input_columns, target_columns):
        self.data = self.__read_file(file_name)
        self.input_columns = input_columns
        self.target_columns = target_columns

    def preprocess_data(self):
        if self.data is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = self.__preprocess_data()

    def get_training_data(self):
        return self.X_train, self.y_train

    def get_testing_data(self):
        return self.X_test, self.y_test

    def get_all_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def __read_file(self, file_name):
        return pd.read_csv(file_name, header=None)

    def __preprocess_data(self):
        input_columns = [i - 1 for i in self.input_columns]
        target_columns = [i - 1 for i in self.target_columns]

        X = self.data.iloc[:, input_columns]
        y = self.data.iloc[:, target_columns]

        non_null_indices = self.__get_non_null_element_index(X, y)
        X = X[non_null_indices]
        y = y[non_null_indices]

        # No need to convert y to int or decrement by 1 for regression

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE_PERCENT, random_state=RANDOM_SEED)

        return self.__convert_dataframes_to_tensors(X_train, X_test, y_train, y_test)

    def __convert_dataframes_to_tensors(self, *dataframes):
        tensors = [torch.from_numpy(df.values).float() for df in dataframes]

        return tensors

    def __get_non_null_element_index(self, X, y):
        return ~X.isnull().any(axis=1) & ~y.isnull().any(axis=1)
