# No_code_DNN( File Upload and Model Training Web Platform)

This web platform allows users to upload CSV files and train a machine learning model using the uploaded data. The platform is built using Flask and provides a simple user interface to upload files and train the model with specified parameters.

## Features

- File upload section for uploading CSV files.
- Training section for specifying model parameters and training the model.
- Display of response messages for successful or failed operations.
- API endpoints for uploading files and training models.

## Usage

1. Navigate to the web page by running the Flask app.
2. In the file upload section, choose a CSV file and click the "Upload" button.
3. In the training section, select the uploaded file from the dropdown menu.
4. Specify input columns, target columns, and the number of nodes in hidden layers.
5. Click the "Train model" button to start the training process.
6. The response messages will display the status of the file upload and model training.

## API Endpoints

- `POST /data`: Uploads a CSV file to the server.
- `GET /data`: Retrieves a list of available CSV files for training.
- `POST /train_model`: Trains the model with the specified parameters from the payload.

## Installation

Clone the repo
```sh
git clone https://github.com/bilet-13/No_code_DNN.git
```
```sh
cd No_code_DNN
```
Activate virtual environment (optional)
For Mac
```sh
python3 -m venv venv
source venv/bin/activate
```
Install necessary packages
```sh
pip install -r requirements
```
Run the src/app.py to start server
```sh
python3 src/app.py
```
The platform will be available at http://localhost:6060/

## Notes

- The platform uses a simple HTML and JavaScript frontend to interact with the Flask backend.
- The `ModelTrainer` and `ModelEvaluator` classes are used for training and evaluating the model, respectively.
- The `get_data` function is used to preprocess the input data for training.
- The `validate_payload` and `is_file_exists` functions are used for payload validation and file existence checks.
- The `is_folder` and `create_folder` functions are used for folder management.
