import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import pathlib
import pickle

NUM_CLASS = 5
DEBUG = False
MODELS_FOLDER_PATH = 'models'
EPOCHS = 100
LEARNING_RATE = 0.001

# Create the models directory if it does not exist
if not os.path.isdir(MODELS_FOLDER_PATH):
    os. makedirs(MODELS_FOLDER_PATH)

class DNN(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size=None):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = NUM_CLASS
        layers = [nn.Linear(input_size, hidden1_size), nn.ReLU()]
        if hidden2_size:
            layers.append(nn.Linear(hidden1_size, hidden2_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden2_size if hidden2_size else hidden1_size, NUM_CLASS))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def parse_args():
    parser = argparse.ArgumentParser(description='Simple DNN for CSV column regression')
    parser.add_argument('--file_name', type=str, required=True, help ='CSV file name')
    parser.add_argument('--input_columns', type=int, nargs='+', required=True, help='Indices for input columns (1-based)')
    parser.add_argument('--target_column', type=int, required=True, help='Index for  target column (1-based)')
    parser.add_argument('--hidden1_size', type=int, required=True, help='Number of nodes in first hidden layer')
    parser.add_argument('--hidden2_size', type=int, default=None, help='Number of nodes in second hidden layer (optional)')
    return parser.parse_args()

def preprocess_data(data, input_columns, target_column):
    input_columns = [i - 1 for i in input_columns]
    target_column = target_column - 1

    X = data.iloc[:, input_columns]
    y = data.iloc[:, target_column]

    non_null_indices = ~X.isnull().any(axis=1) & ~np.isnan(y)
    X = X[non_null_indices]
    y = y[non_null_indices]

    y = y.values.astype(int) - 1
    # print(type(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        if torch.isnan(loss):
            continue

        loss.backward()
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
        'hidden2_size': model.hidden2_size
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_info, f)


def load_model(filepath):
    # Load both the model state and the model architecture
    with open(filepath, 'rb') as f:
        model_info = pickle.load(f)
    
    # Reconstruct the model using the saved architecture
    model = DNN(model_info['input_size'],
                model_info['hidden1_size'],
                model_info['hidden2_size'])
    model.load_state_dict(model_info['state_dict'])
    
    return model

def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(X_test)
        #print(outputs)
        predictions = torch.argmax(outputs, dim=1)
        #print(predictions)
        accuracy = (predictions == y_test).sum().item() / len(y_test)
    return accuracy

def main():
    args = parse_args()

    data = pd.read_csv(args.file_name, header=None)

    X_train, X_test, y_train, y_test = preprocess_data(data, args.input_columns, args.target_column)
    
    input_size = len(args.input_columns)
    model = DNN(input_size, args.hidden1_size, args.hidden2_size)

    # Convert DataFrames to Tensors
    X_train = torch.from_numpy(X_train.values).float()
    X_test = torch.from_numpy(X_test.values).float()

    y_train = torch.LongTensor(y_train)

    train_model(model, X_train, y_train)

    y_test = torch.LongTensor(y_test)

    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {accuracy}')

    file_name = f'hidden1_size_{args.hidden1_size}_and_hidden2_size_{args.hidden2_size}.pt'
    file_path = pathlib.Path(MODELS_FOLDER_PATH, file_name)
    save_model(model, file_path)
    print(f'save model as {file_name} in {MODELS_FOLDER_PATH} folder')

    loaded_model = load_model(file_path)
    accuracy = evaluate_model(loaded_model, X_test, y_test)
    #print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    main()
