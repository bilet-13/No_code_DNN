from simple_DNN import DNN
import torch
import torch.nn as nn
import pathlib
import pickle

DEBUG = False
EPOCHS = 500
LEARNING_RATE = 0.001

class ModelTrainer():

    def __init__(self):
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.rmse_loss = None

    def init_model(self, input_size, hidden1_size, hidden2_size, output_size):
        self.model = DNN(input_size, hidden1_size, hidden2_size, output_size)


    def train_model(self, X_train, y_train):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        rmse_loss = 0

        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            if torch.isnan(loss):
                continue

            rmse_loss = torch.sqrt(loss)

            if DEBUG:
                print(rmse_loss)

            rmse_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            optimizer.step()

        self.rmse_loss = rmse_loss

    def get_rmse(self):
        return self.rmse_loss.item()

    def get_model(self):
        return self.model
    
    def save_model(self, folder_path):
    # Save both the model state and the model architecture
        model_info = {
            'state_dict': self.model.state_dict(),
            'input_size': self.model.input_size,
            'hidden1_size': self.model.hidden1_size,
            'hidden2_size': self.model.hidden2_size,
            'output_size': self.model.output_size,
            'model_name': self.model.get_model_name()
        }
        file_name = self.model.get_model_name() + '.pt'
        file_path = pathlib.Path(folder_path, file_name)

        with open(file_path, 'wb') as f:
            pickle.dump(model_info, f)
