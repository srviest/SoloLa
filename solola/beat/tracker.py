
from pathlib import Path
import madmom
import numpy as np
import sys
import subprocess
import os


def downbeat_tracking(audioPath):
    proc = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar=[
        4, 4], fps=100)
    act = madmom.features.RNNDownBeatProcessor()(audioPath)
    # print(proc(act))
    downbeats = proc(act)
    return downbeats


def beat_tracking(fp):
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(fp)
    # print(proc(act))
    beats = proc(act)
    return beats

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def write_file(output_path, content=""):
    fp = Path(output_path)
    print()

    parent_dir_path = fp.parents[0]
    if not parent_dir_path.exists() and not parent_dir_path.is_dir():
        print('not exist')
        Path.mkdir(parent_dir_path)

    file = open(output_path, "w+")
    file.write(content)

    print('done')


def track(args):
    audio_path = Path(args.audio_path)
    if audio_path.suffix != '.wav' and audio_path.suffix != '.mp3':
        print("unsupported audio type (only support mp3,wav)")
        exit()

    if audio_path.suffix == '.mp3' or audio_path.suffix == '.MP3':
        print("[Info] Automatically convert .mp3 to .wav by ffmpeg")
        subprocess.run(["ffmpeg", "-y", "-i", "{}/{}".format(audio_path.parent, audio_path.name),
                        "{}/{}.wav".format(audio_path.parent, audio_path.stem)])

    audio_path = "{}/{}.wav".format(audio_path.parent, audio_path.stem)

    # beat
    print("[Info] Start beat_tracking....")
    beats = beat_tracking(audio_path)
    beats_string = ''
    for i, v in enumerate(beats):
        beats_string += "{},{}\n".format(i, v)
    write_file("{}/beats.txt".format(args.output_dir_path), beats_string)
    print("[Info] Finish beat_tracking")

    # downbeat
    print("[Info] Start downbeat_tracking")
    downbeats = downbeat_tracking(audio_path)
    downbeats_string = ''
    for i, v in enumerate(downbeats):
        downbeats_string += "{},{}\n".format(v[0], v[1])
    write_file("{}/downbeats.txt".format(args.output_dir_path),
               downbeats_string)
    print("[Info] Finish downbeat_tracking")


class TrackerArgument:
    def __init__(self):
        pass
