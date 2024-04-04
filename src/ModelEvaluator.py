import pickle
from simple_DNN import DNN
import torch
import torch.nn as nn

DEBUG = True

class ModelEvaluator():

    def __init__(self):
        self.model = None
        self.rmse = None

    def load_model(self, filepath):
        # Load both the model state and the model architecture
        with open(filepath, 'rb') as f:
            model_info = pickle.load(f)

        # Reconstruct the model using the saved architecture
        self.model = DNN(
            model_info['input_size'],
            model_info['hidden1_size'],
            model_info['hidden2_size'],
            model_info['output_size']
        )
        self.model.load_state_dict(model_info['state_dict'])

    def set_model(self, model):
        self.model = model
        
    def evaluate_model(self, X_test, y_test):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.model(X_test)
            print(outputs.shape)
            if DEBUG:
                print(outputs)
            mse_loss = nn.MSELoss()(outputs, y_test)  # Compute MSE using PyTorch
            # rmse_losses = torch.sqrt(nn.MSELoss(reduction='none')(outputs, y_test))  
            # Compute RMSE for each column
            rmse_loss = torch.sqrt(mse_loss)  # Compute RMSE
        return rmse_loss.item()  # Return the RMSE as a Python float