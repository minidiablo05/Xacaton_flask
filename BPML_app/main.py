from flask import Flask, request, render_template, jsonify
from config import Config
import json
import soundfile as sf
import io
from vosk import Model, KaldiRecognizer
import subprocess
import os

app = Flask(__name__)

app.config.from_object(Config)

# Конфигурация
model = Model(Config.MODEL_PATH_VOSK)


@app.route('/', methods=['GET', 'POST'])
def transcribe_audio():
    '''Основная функция обработки работы страницы сайта.'''

    if request.method == 'GET':
        return render_template('transcribe_audio.html')

    if request.method == 'POST':
        # Обработка аудиофайла
        if 'audio' in request.files:
            audio_file = request.files['audio']

            # Читаем содержимое файла в память
            audio_data = audio_file.read()

            # Преобразуем в BytesIO (как файловый объект)
            audio_bytes = io.BytesIO(audio_data)

            try:
                # Читаем WAV-файл
                wav_data, sample_rate = sf.read(audio_bytes, dtype='int16')
                recognizer = KaldiRecognizer(model, sample_rate)
                recognizer.AcceptWaveform(wav_data.tobytes())

                # Получаем результат
                result = json.loads(recognizer.FinalResult())
                text = result.get("text", "")
                response_data = {
                    'response': text
                }
            except Exception as e:
                response_data = {
                    'response': f'Сбой программы: {str(e)}'
                }

            return jsonify(response_data)

        # Обработка текста
        elif request.json:


            try:
                received_text = request.json.get('text')
                if not received_text:
                    return jsonify({'error': 'Missing "text"'}), 400

                # python_interpreter = os.path.join('.venv', 'Scripts', 'python.exe')
                # main_script_path = os.path.join(
                #     'Xacaton-1-process-visualizer-reborn-BPMN-develop',
                #     'process-visualizer',
                #     'src',
                #     'main.py'
                # )

                result = subprocess.run(
                    [Config.python_interpreter,
                     Config.ml_main_script_path,
                     "-t",
                     received_text],
                    capture_output=True,
                    text=True
                )

                print(result.stderr)
                print(result.stdout)

                response_data = {
                    'response': result.stderr
                }

                return jsonify(response_data)

            except json.JSONDecodeError:
                return jsonify({'error': 'Invalid JSON'}), 400
            except FileNotFoundError:
                return jsonify({'error': 'Script not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid request'}), 400


if __name__ == '__main__':
    app.run(debug=True)
