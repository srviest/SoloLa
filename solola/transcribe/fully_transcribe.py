import os
import sys
root_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_path)

from pathlib import Path
from MusicXMLSynthesizer.utils import parse_notes_meta_to_list, write_file
from MusicXMLSynthesizer.Synthesizer import Synthesizer
from solola.beat.tracker import track, TrackerArgument
from solola.Solola import Solola

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
===================================================================
Script for full transcription(audio => mzxml).
===================================================================
    """)
    parser.add_argument(
        '-s',
        '--audio_path',
        type=str,
        help="""
        The path of input audio file e.g. input/slide.mp3
        """,
        default="inputs/slide.mp3"
    )
    parser.add_argument(
        '-o',
        '--output_dir_path',
        type=str,
        help="""
        The path of beats/downbeat output directory e.g. outputs/slide
        """,
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

# temp usage: will be removed after musicXML-Synthesizer fixing bugs
def write_xml_to_file(output_path, xml):
    fp = Path(output_path)
    print()

    parent_dir_path = fp.parents[0]
    if not parent_dir_path.exists() and not parent_dir_path.is_dir():
        print('not exist')
        Path.mkdir(parent_dir_path)

    with open(output_path, "w+") as file:
        file.write(xml)
        print('done')
    

if __name__ == '__main__':
    args = parse_args()
    audio_path = args.audio_path
    output_dir_path = args.output_dir_path

    track_beats(audio_path, output_dir_path)
    transcribe_guitar_tech(audio_path, output_dir_path)

    audio_file_name = Path(audio_path).stem
    techs_and_notes_list = parse_notes_meta_to_list(
        "{}/{}/FinalNotes.txt".format(output_dir_path, audio_file_name))
    beats_list = parse_notes_meta_to_list(
        "{}/beats.txt".format(output_dir_path))
    downbeats_list = parse_notes_meta_to_list(
        "{}/downbeats.txt".format(output_dir_path))

    print(techs_and_notes_list)
    print(downbeats_list)
    print(beats_list)

    synthesizer = Synthesizer()
    synthesizer.save(techs_and_notes_list, downbeats_list, beats_list)
    mzxml = synthesizer.execute()
    write_xml_to_file("{}/{}.mzxml".format(output_dir_path, audio_file_name), mzxml)
