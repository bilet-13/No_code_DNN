import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import pathlib
import pickle

DEBUG = False
NUM_CLASS = 5
TEST_SIZE_PERCENT = 0.3
RANDOM_SEED = 42
EPOCHS = 100
LEARNING_RATE = 0.001
MODELS_FOLDER_PATH = 'models'

# Create the models directory if it does not exist
if not os.path.isdir(MODELS_FOLDER_PATH):
    os. makedirs(MODELS_FOLDER_PATH)


class DNN(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        layers = [nn.Linear(input_size, hidden1_size), nn.ReLU()]
        if hidden2_size:
            layers.append(nn.Linear(hidden1_size, hidden2_size))
            layers.append(nn.ReLU())
        # Specify output size
        layers.append(
            nn.Linear(hidden2_size if hidden2_size else hidden1_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple DNN for CSV column regression')
    parser.add_argument('--file_name', type=str,
                        required=True, help='CSV file name')
    parser.add_argument('--input_columns', type=int, nargs='+',
                        required=True, help='Indices for input columns (1-based)')
    parser.add_argument('--target_columns', type=int, nargs='+',
                        required=True, help='Indices for  target column (1-based)')
    parser.add_argument('--hidden1_size', type=int, required=True,
                        help='Number of nodes in first hidden layer')
    parser.add_argument('--hidden2_size', type=int, default=None,
                        help='Number of nodes in second hidden layer (optional)')
    return parser.parse_args()


def get_non_null_element_index(X, y):
    return ~X.isnull().any(axis=1) & ~y.isnull().any(axis=1)


def preprocess_data(data, input_columns, target_columns):
    input_columns = [i - 1 for i in input_columns]
    target_columns = [i - 1 for i in target_columns]

    X = data.iloc[:, input_columns]
    y = data.iloc[:, target_columns]

    non_null_indices = get_non_null_element_index(X, y)
    X = X[non_null_indices]
    y = y[non_null_indices]

    # No need to convert y to int or decrement by 1 for regression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE_PERCENT, random_state=RANDOM_SEED)

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        if torch.isnan(loss):
            continue

        rmse_loss = torch.sqrt(loss)

        if DEBUG:
            print(rmse_loss)

        rmse_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        if DEBUG:
            print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}')


def save_model(model, filepath):
    # Save both the model state and the model architecture
    model_info = {
        'state_dict': model.state_dict(),
        'input_size': model.input_size,
        'hidden1_size': model.hidden1_size,
        'hidden2_size': model.hidden2_size,
        'output_size': model.output_size
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_info, f)


def load_model(filepath):
    # Load both the model state and the model architecture
    with open(filepath, 'rb') as f:
        model_info = pickle.load(f)

    # Reconstruct the model using the saved architecture
    model = DNN(
        model_info['input_size'],
        model_info['hidden1_size'],
        model_info['hidden2_size'],
        model_info['output_size']
    )
    model.load_state_dict(model_info['state_dict'])

    return model


def evaluate_model(model, X_test, y_test):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X_test)

        if DEBUG:
            print(outputs)
        mse_loss = nn.MSELoss()(outputs, y_test)  # Compute MSE using PyTorch
        rmse_loss = torch.sqrt(mse_loss)  # Compute RMSE
    return rmse_loss.item()  # Return the RMSE as a Python float


def main():
    args = parse_args()

    data = pd.read_csv(args.file_name, header=None)
    # Adjust to handle multiple target columns
    X_train, X_test, y_train, y_test = preprocess_data(
        data, args.input_columns, args.target_columns)

    input_size = len(args.input_columns)
    # Get the output size from arguments
    output_size = len(args.target_columns)
    model = DNN(input_size, args.hidden1_size, args.hidden2_size, output_size)

    # Convert DataFrames to Tensors
    X_train = torch.from_numpy(X_train.values).float()
    X_test = torch.from_numpy(X_test.values).float()

    # Convert target to float and reshape for multiple outputs
    y_train = torch.from_numpy(y_train.values).float()
    y_test = torch.from_numpy(y_test.values).float()

    train_model(model, X_train, y_train)

    rmse = evaluate_model(model, X_test, y_test)
    print(f'RMSE: {rmse}')


if __name__ == '__main__':
    main()
