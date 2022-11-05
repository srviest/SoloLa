# add root_path 
import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_path)

# from flask_restful import Api
# from resources.TranscriptionTask import TranscriptionTask, TranscriptionTaskList
from flask import Flask, jsonify, render_template, request
app = Flask(__name__)
# api = Api(app)
# api.add_resource(TranscriptionTask, '/task/<string:id>')
# api.add_resource(TranscriptionTaskList, '/tasks')

# solola transcribe code
from solola.Solola import Solola

# beat tracker
from solola.beat.tracker import track, TrackerArgument

@app.route("/")
def index():
    return jsonify(version="v0.1.0")

@app.route('/transcription', methods=['POST'])
def transcription():
    # audio_id
    if request.is_json:
        # print('raw:', request.data)
        data = request.json
        if isDemo(data['audio_id']):
            args = TrackerArgument()
            args.audio_path = 'services/input/slide.mp3'
            args.output_dir_path = 'services/output/slide'
            track(args)

    # audio_file_path = './services/mock/audio/music.mp3'
    # output_dir = 'services/output'
    # default_asc_model = 'models/cnn_normmc/ascending.npz'
    # default_desc_model = 'models/cnn_normmc/descending.npz'

    # runner = Solola(audio_fp=audio_file_path, asc_model_fp=default_asc_model, desc_model_fp=default_desc_model, output_dir=output_dir)
    # runner.extract_melody()
    # runner.transcribe()

    return jsonify(version="track beat")

def isDemo(audio_id):
    return audio_id == 0;

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')