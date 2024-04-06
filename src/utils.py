import pathlib

def validate_payload(payload):
    required_fields = ['file_name', 'input_columns', 'target_columns', 'hidden1_size', 'hidden2_size']
    if not all(field in payload for field in required_fields):
        return False
    if not all(isinstance(payload[field], (int, list)) for field in ['hidden1_size', 'hidden2_size']):
        return False
    return True


def is_file_in_folder(file_name, folder_path):
    file_path = pathlib.Path(folder_path) / file_name
    return file_path.is_file()

def is_folder(folder_path):
   folder_path = pathlib.Path(folder_path)
   return folder_path.is_dir()

def create_folder(folder_path):
   folder_path = pathlib.Path(folder_path)
   folder_path.mkdir(parents=True, exist_ok=True)
   return None
   



