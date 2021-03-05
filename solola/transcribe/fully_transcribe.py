import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_path)
from solola.Solola import Solola
from solola.beat.tracker import track, TrackerArgument

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description=
    """
===================================================================
Script for full transcription(audio => mzxml).
===================================================================
    """)
    parser.add_argument(
        '-s', 
        '--audio_path', 
        type=str, 
        help="absolute path of audio file", 
        default="inputs/slide.mp3"
    )
    parser.add_argument(
        '-o', 
        '--output_dir_path', 
        type=str, 
        help="absolute path of beats/downbeat output directory", 
        default="outputs/slide"
    )
    
    return parser.parse_args()
    
def track_beats(audio_path, output_dir_path):
    args = TrackerArgument()
    args.audio_path = audio_path
    args.output_dir_path = output_dir_path
    track(args)

def transcribe_guitar_tech(audio_path, output_dir_path):
    default_asc_model = 'models/cnn_normmc/ascending.npz'
    default_desc_model = 'models/cnn_normmc/descending.npz'

    runner = Solola(
        audio_fp=audio_path, 
        asc_model_fp=default_asc_model, 
        desc_model_fp=default_desc_model, 
        output_dir=output_dir_path
    )
    runner.extract_melody()
    runner.transcribe()

if __name__ == '__main__':
    args = parse_args()
    audio_path = args.audio_path
    output_dir_path = args.output_dir_path
    
    track_beats(audio_path, output_dir_path)
    transcribe_guitar_tech(audio_path, output_dir_path)