from lib import transcribe
import librosa as rosa
import numpy as np
from guitar_trans.contour import *
# from guitar_trans.technique import *
from guitar_trans.evaluation import evaluation_note, evaluation_esn, evaluation_ts
from os import path, sep, makedirs

# melody components
from melody.melody_extraction import extract_melody


class Solola:
    def __init__(self, audio_fp, asc_model_fp, desc_model_fp, output_dir, mc_fp=None):
        self.audio_fp = audio_fp
        self.asc_model_fp = asc_model_fp
        self.desc_model_fp = desc_model_fp
        self.save_dir = output_dir
        self.audio_fn = path.splitext(path.basename(audio_fp))[0]
        self.save_dir = path.join(output_dir, self.audio_fn)
        self.mc_fp = mc_fp
        self.notes = None
        self.audio = None
        self.melody = None

        # input/output
        print("audio_fp:", self.audio_fp)
        print("save_dir:", self.save_dir)

    def extract_melody(self):
        # Setup model
        if self.mc_fp is None:
            mc, mc_midi = extract_melody(self.audio_fp, self.save_dir)
        else:
            mc_midi = np.loadtxt(self.mc_fp)

        # Prepare transcribed input
        audio, sr = rosa.load(self.audio_fp, sr=None, mono=True)
        self.audio = audio
        self.melody = Contour(0, mc_midi)
        return self.melody

    def transcribe(self):
        if self.audio is None or self.melody is None:
            print('audio or melody is not set yet')
            return

        if self.asc_model_fp is None or self.desc_model_fp is None:
            print('model is not set properly')

        self.notes = transcribe(self.audio, self.melody, self.asc_model_fp,
                                self.desc_model_fp, self.save_dir, self.audio_fn)
        return self.notes

    def evaluate(self, eval_note):
        if self.notes is None:
            print('The result notes does not exist')
            return
        if eval_note is not None:
            sg = Song(name=self.audio_fn)
            sg.load_esn_list(eval_note)
            evaluation_note(sg.es_note_list, self.notes, self.save_dir,
                            self.audio_fn, string='evaluate notes')
            evaluation_esn(sg.es_note_list, self.notes, self.save_dir,
                           self.audio_fn, string='evaluate esn')
        pass

    def synthesizeMZXML(self):
        # generate mzxml
        return 'mzxml'
