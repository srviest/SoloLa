# add root_path 
import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_path)

# from flask_restful import Api
# from resources.TranscriptionTask import TranscriptionTask, TranscriptionTaskList
from flask import Flask, jsonify, render_template
app = Flask(__name__)
# api = Api(app)
# api.add_resource(TranscriptionTask, '/task/<string:id>')
# api.add_resource(TranscriptionTaskList, '/tasks')

# solola transcribe code
# import main
from solola.Solola import Solola

@app.route("/")
def index():
    return jsonify(version="v0.1.0")

@app.route('/transcription', methods=['POST'])
def transcription():
    audio_file_path = './services/mock/audio/music.mp3'
    output_dir = 'services/output'
    default_asc_model = 'models/cnn_normmc/ascending.npz'
    default_desc_model = 'models/cnn_normmc/descending.npz'
    # main.main(audio_file_path, default_asc_model, default_desc_model, output_dir)
    runner = Solola(audio_fp=audio_file_path, asc_model_fp=default_asc_model, desc_model_fp=default_desc_model, output_dir=output_dir)
    runner.extract_melody()
    runner.transcribe()

    return jsonify(version="v0.1.0")



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')