from flask_restful import Resource, abort, reqparse

# THIS need to be access by solola
tasks = {
    '0': {
        'status': 'EMPTY'
    }
}

def abort_if_not_found(id):
    if id not in tasks:
        abort(404, message="Task {} doesn't exist".format(id))

parser = reqparse.RequestParser()
parser.add_argument('begin')
parser.add_argument('end')
parser.add_argument('ythash')

class TranscriptionTask(Resource):
    def get(self, id):
        abort(id)
        return tasks[id]

    # SHOULD WE UPDATE transcription task during process?
    # def put(self, id):
    #     args = parser.parse_args()
    #     tasks[id] = {
    #         'status': 'CREATED',
    #         'begin': args['begin'],
    #         'end': args['end'],
    #         'yt_hash': args['yt_hash']
    #     }
    #     return tasks[id], 204

class TranscriptionTaskList(Resource):
    def get(self):
        return tasks

    def post(self):
        args = parser.parse_args()
        id = int(max(tasks.keys())) + 1
        
        # TODO: Solola transcribe

        tasks[id] = {
            'status': 'CREATED',
            'begin': args['begin'],
            'end': args['end'],
            'ythash': args['ythash']
        }
        return tasks[id], 201