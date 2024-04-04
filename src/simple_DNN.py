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
EPOCHS = 500
LEARNING_RATE = 0.001
MODELS_FOLDER_PATH = 'models'


class DNN(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.model_name = self.__init_model_name()

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

    def get_model_name(self):
        return self.model_name
    
    def __init_model_name(self):
        name = f''
        name += f'input_size{self.input_size}_hidden1_size{self.hidden1_size}'
        name += f'' if self.hidden2_size is None else f'_hidden2_size{self.hidden2_size}'
        name += f'_output_size{self.output_size}'
        return name
    

def main():
    return None

if __name__ == '__main__':
    main()
