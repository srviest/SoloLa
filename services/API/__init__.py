# from flask import Flask, jsonify
# from flask_socketio import SocketIO

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app)

# @app.route("/")
# def hello_world():
#     return jsonify(hello="world")

# @app.route('/api/v1/transcription/',  methods=['POST'])
# def transcribe():
#     # create a socket to real-time response progress
#     # download youtube video
#     # mp4 tp wav
#     # run solola
#     #  
#     return jsonify(hello="world")

# if __name__ == '__main__':
#     socketio.run(app)