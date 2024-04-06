import pathlib
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer

def get_training_data(file_name, input_columns, target_columns):
    data_processor = DataProcessor()
    data_processor.load_data(file_name, input_columns, target_columns)
    data_processor.preprocess_data()

    return data_processor.get_training_data()

def get_data(file_name, input_columns, target_columns):
    data_processor = DataProcessor()
    data_processor.load_data(file_name, input_columns, target_columns)
    data_processor.preprocess_data()
   
    return data_processor.get_all_data()