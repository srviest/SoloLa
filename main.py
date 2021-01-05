from lib import transcribe
import librosa as rosa
import numpy as np
from guitar_trans.contour import *
# from guitar_trans.technique import *
from guitar_trans.evaluation import evaluation_note, evaluation_esn, evaluation_ts
from os import path, sep, makedirs

# melody components
from melody.melody_extraction import extract_melody


def main(audio_fp, asc_model_fp, desc_model_fp, output_dir, mc_fp=None, eval_note=None, eval_ts=None):
    # input/output
    audio_fn = path.splitext(path.basename(audio_fp))[0]
    save_dir = path.join(output_dir, audio_fn)
    print("audio_fp:", audio_fp)
    # Setup model 
    if mc_fp is None:
        mc, mc_midi = extract_melody(audio_fp, save_dir)
    else:
        mc_midi = np.loadtxt(mc_fp)

    # Prepare transcribed input
    audio, sr = rosa.load(audio_fp, sr=None, mono=True)
    melody = Contour(0, mc_midi)

    notes = transcribe(audio, melody, asc_model_fp, desc_model_fp, save_dir, audio_fn)
    
    if eval_note is not None:
        sg = Song(name=audio_fn)
        sg.load_esn_list(eval_note)
        evaluation_note(sg.es_note_list, notes, save_dir, audio_fn, string='evaluate notes')
        evaluation_esn(sg.es_note_list, notes, save_dir, audio_fn, string='evaluate esn')

    if eval_ts is not None:
        ans_list = np.loadtxt(eval_ts)
        # TODO

def parser():
    import argparse
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description=
    """
===================================================================
Script for transcribing a song.
===================================================================
    """)
    p.add_argument('audio_fp', type=str, metavar='audio_fp',
                    help='The filepath of the audio to be transcribed.')
    p.add_argument('-a', '--asc_model_fp', type=str, metavar='asc_model_fp', default='models/cnn_normmc/ascending.npz',
                    help='The name of the ascending model.')
    p.add_argument('-d', '--desc_model_fp', type=str, metavar='desc_model_fp', default='models/cnn_normmc/descending.npz',
                    help='The name of the descending model.')
    p.add_argument('-o', '--output_dir', type=str, metavar='output_dir', default='outputs',
                    help='The output directory.')
    p.add_argument('-m', '--melody_contour', type=str, default=None, 
                    help='The filepath of melody contour.')
    p.add_argument('-e', '--evaluate', type=str, default=None, 
                    help='The filepath of answer file.')
    return p.parse_args()

if __name__ == '__main__':
    args = parser()
    main(args.audio_fp, args.asc_model_fp, args.desc_model_fp, 
         args.output_dir, args.melody_contour, args.evaluate)
