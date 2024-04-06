# app.py
from flask import Flask, request, jsonify, render_template
from ModelTrainer import ModelTrainer
from ModelEvaluator import ModelEvaluator
from API_function import get_data
from utils import validate_payload, is_file_exists
from utils import is_folder, create_folder
import pathlib

MODELS_FOLDER = 'models'
TRAINING_FILES_FOLDER = 'data'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = TRAINING_FILES_FOLDER


@app.route('/')
def get_web_page():
    return render_template('platform.html')

@app.route('/data', methods=['POST'])
def upload():
    file = request.files.get('file')  # Use the get method to avoid KeyError
    if file:  # Check if the file is present
        data_folder_path = pathlib.Path(TRAINING_FILES_FOLDER)
        file.save(data_folder_path.joinpath(file.filename))
        return jsonify({'success': True}), 200
    else:
        return jsonify({'error': 'No file part'}), 400


@app.route ('/data', methods=['GET'])
def get_files():
    data_folder_path = pathlib.Path(TRAINING_FILES_FOLDER)
    files = [file_path.name for file_path in data_folder_path.glob('*.csv')]
    return jsonify({'files': files}), 200

@app.route('/train_model', methods=['POST'])
def train_model():
    payload = request.get_json()
    print(payload)
    if not validate_payload(payload):
        return jsonify({"error": "Invalid payload"}), 400

    file_name = payload['file_name']
    input_columns = payload['input_columns']
    target_columns = payload['target_columns']
    hidden1_size = payload['hidden1_size']
    hidden2_size = payload ['hidden2_size']

    if not is_file_exists(file_name):
        return jsonify({"error": f"File not found: {file_name}"}), 400

    try:
        hidden1_size = None if hidden1_size <= 0 else hidden1_size
        hidden2_size = None if hidden2_size <= 0 else hidden2_size
        X_train, y_train, X_test, y_text = get_data(file_name, input_columns, target_columns)

        model_trainer = ModelTrainer()
        model_trainer.init_model(
            len(input_columns ), hidden1_size, hidden2_size, len(target_columns))
        model_trainer.train_model(X_train, y_train)
        # model_trainer.save_model(MODELS_FOLDER)

        training_rmse_loss = model_trainer.get_rmse()

        model_evaluator = ModelEvaluator()
        model_evaluator.set_model(model_trainer.get_model())
        evaluation_rmse = model_evaluator.evaluate_model(X_test, y_text)

        return jsonify({"Training data RMSE": training_rmse_loss, "Testing data RMSE": evaluation_rmse}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    if not is_folder(TRAINING_FILES_FOLDER):
        create_folder(TRAINING_FILES_FOLDER)
    app.run(debug=True, port=6060)
