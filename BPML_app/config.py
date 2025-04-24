import os
# print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASE_DIR)
# print(os.path.join('.venv', 'Scripts', 'python.exe'))


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH_VOSK = os.path.join(BASE_DIR, 'model', 'vosk-model-small-ru-0.22')
    UPLOAD_FOLDER = 'uploads'
    python_interpreter = os.path.join(BASE_DIR, '.venv', 'Scripts', 'python.exe')
    ml_main_script_path = os.path.join(
        BASE_DIR,
        'Xacaton-1-process-visualizer-reborn-BPMN-develop',
        'process-visualizer',
        'src',
        'main.py'
        )
