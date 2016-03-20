#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2015/10/13

Guitar playing technique baseline experiment (S1-S4).
----------------------------------------------------------------------
Baseline playing technique experiment pipeline:
----------------------------------------------------------------------
S1 Melody extraction
    S1.1 Extract melody contour
    S1.3 Find continuously ascending/descending (CAD) F0 sequence patterns
    S1.4 Find intersection of note and pattern (Candidate selection of {bend,slide,pull-off,hammer-on,normal})
----------------------------------------------------------------------
S2 Note tracking
----------------------------------------------------------------------
S3 Find CAD F0 sequence pattern
----------------------------------------------------------------------
S4 Candidate selection
----------------------------------------------------------------------
S5 Feature extraction
----------------------------------------------------------------------
S6 Classifier training
----------------------------------------------------------------------
"""
import glob, os, sys
sys.path.append('/Users/Frank/Documents/Code/Python')
sys.path.append('/Users/Frank/Documents/Code/Python/libsvm-3.18/python')
import numpy as np
from scipy.io import wavfile
import math
from essentia.standard import *
import subprocess as subp
# import matplotlib.pyplot as plt
from scipy.io import wavfile
import operator
from svmutil import *
from PTNoteTransitionOverlapEval import *
from io_tool import audio2wave


def parse_input_files(args):
    from os.path import basename, isdir
    import glob
    files = []
    for i in args.input_files:
        # check what we have (file/path)
        if isdir(i):
            # use all files in the given path
            if args.k:
                all_files = glob.glob(i + '/*.wav')
                for kw in args.k:
                    for w in all_files:
                        if basename(w).find(kw)!=-1:
                            files.append(w)
            else:
                files = glob.glob(i + '/*.wav')           
        else:
            # file was given, append to list
            files.append(i)
    print 'Inout files: '
    for f in files: print '  ', f
    return files

def parser():
    """
    Parses the command line arguments.

    :param lgd:       use local group delay weighting by default
    :param threshold: default value for threshold

    """
    import argparse
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters, the software S1 Extract melody contour,
     track notes and timestmaps of intersection of ad continuous pitch sequence
     inthe given files, the pipeline is as follows,

        S1.1 Extract melody contour
        S1.2 Note tracking
        S1.3 Find continuously ascending/descending (CAD) F0 sequence patterns
        S1.4 Find intersection of note and pattern 
             (Candidate selection of {bend,slide,pull-off,hammer-on,normal})
    """)
    # general options
    p.add_argument('input_files', type=str, metavar='input_files', nargs='+',
                   help='files to be processed')
    p.add_argument('output_dir', type=str, metavar='output_dir', nargs=1,
                   help='output directory.')
    p.add_argument('-k','--keyword', nargs='+',dest = 'k', help="key word in files to be processed.", default=None)
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2014-11-02)')
    # parse arguments
    args = p.parse_args()

    # return args
    return args
    

def main(args):
    print 'Running Baseline Playing Technique S1, S2, S3 and S4.'
    file_dir = '/Users/Frank/Documents/Code/Database/test'
    
    # parse and list files to be processed
    files = parse_input_files(args)
    
    
        
    # create result directory
    cwd = os.getcwd()
    output_dir = os.path.join(cwd,'BaselinePT')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print 'Output directory:'
    print '  ', output_dir

    # processing
    for f in files:
        # parse file name and extension
        ext = os.path.basename(f).split('.')[-1]
        name = os.path.basename(f).split('.')[0]

        """
        S1 Extract melody contour
        """
        # create result directory
        S1_melody_dir = os.path.join(output_dir,'S1_melody')
        if not os.path.exists(S1_melody_dir):
            os.makedirs(S1_melody_dir)
        # parsmeters for MELODIA
        m_frameSize = 2048
        m_hopSize = 256
        m_samplerate = 44100
        sr = 44100
        minf0 = 77
        maxf0 = 1400
        guessUnvoiced = True
        voiceVibrato = True
        voicingTolerance = 0.2
        binResolution = 10
        buffer_size = 3
        minDuration = 100
        harmonicWeight = 0.8
        # S1.1 estimate predominant melody
        pcm = PredominantMelody(harmonicWeight = harmonicWeight, minDuration = minDuration, 
            binResolution = binResolution, guessUnvoiced=guessUnvoiced, frameSize=m_frameSize, 
            hopSize=m_hopSize, maxFrequency = maxf0, minFrequency = minf0, sampleRate = sr, 
            voiceVibrato = voiceVibrato, voicingTolerance = voicingTolerance)
        audio = audio2wave(f)
        melody_contour, pitchConfidence = pcm(audio)
        # save result: raw melody contour
        np.savetxt(S1_melody_dir+os.sep+name+'.melody'+'.txt',melody_contour, fmt='%s')
        # S1.2 convert Hz to MIDI scale
        melody_contour = hertz2midi(melody_contour)
        np.savetxt(S1_melody_dir+os.sep+name+'.melody.MIDI'+'.txt',melody_contour, fmt='%s')
        # S1.3 moving averaging filtering
        melody_contour = mean_filter(melody_contour,kernel_size = 5)
        # save result: MIDI-scaled moving-averaged melody contour
        np.savetxt(S1_melody_dir+os.sep+name+'.melody.MIDI.smooth'+'.txt',melody_contour, fmt='%s')

        """
        S2 Note tracking
        """

        # create result directory
        S2_note_dir = os.path.join(output_dir,'S2_note')
        if not os.path.exists(S2_note_dir):
            os.makedirs(S2_note_dir)
        # S2.1 recognize note event in estimated melody contour
        NoteRecognizer_path = '/Users/Frank/Documents/Code/C++/Note_recognizer/MOLODIA_HMM/NoteRecognizer'
        input_path = os.path.join(S1_melody_dir,name+'.melody'+'.txt')
        command = [NoteRecognizer_path, input_path]
        pipe = subp.Popen(command, stdout=subp.PIPE, startupinfo=None)
        note_string = pipe.stdout.read()
        note = []
        for line in note_string.splitlines():
          note.append(np.fromstring(line, dtype="float32", sep=' '))
        note = np.asarray(note)
        # save result: note event
        np.savetxt(S2_note_dir+os.sep+name+'.note'+'.txt',note, fmt='%s')
        # S2.2 note pruning
        note = note_pruning(note, 0.1)
        # save result: prunied note event
        np.savetxt(S2_note_dir+os.sep+name+'.note.prune.txt',note, fmt='%s')


        """
        S3 Find continuously ascending/descending (CAD) F0 sequence patterns
        """
        # create result directory
        S3_CAD_dir = os.path.join(output_dir,'S3_CAD')
        if not os.path.exists(S3_CAD_dir):
            os.makedirs(S3_CAD_dir)
        # S3.1 find continuously ascending/descending (CAD) F0 sequence patterns
        ascending_pattern, ascending_pitch_contour = continuously_ascending_descending_pattern(melody_contour,direction='up',MinLastingDuration=0.05, MaxPitchDifference = 3.8, MinPitchDifference=0.8,hop=m_hopSize,fs=sr)
        descending_pattern, descending_pitch_contour = continuously_ascending_descending_pattern(melody_contour,direction='down',MinLastingDuration=0.05, MaxPitchDifference = 3.8, MinPitchDifference=0.8,hop=m_hopSize,fs=sr)
        # save result: CAD F0 sequence pattern
        np.savetxt(S3_CAD_dir+os.sep+name+'.pattern.ascending'+'.txt',ascending_pattern, fmt='%s')
        np.savetxt(S3_CAD_dir+os.sep+name+'.pitch_contour.ascending'+'.txt',ascending_pitch_contour, fmt='%s')
        np.savetxt(S3_CAD_dir+os.sep+name+'.pattern.descending'+'.txt',descending_pattern, fmt='%s')
        np.savetxt(S3_CAD_dir+os.sep+name+'.pitch_contour.descending'+'.txt',descending_pitch_contour, fmt='%s')

        """
        S4 Candidate selection
        Find intersection of note and pattern of {bend,slide,pull-off,hammer-on,normal})
        """
        # create result directory
        S4_candidate_dir = os.path.join(output_dir,'S4_candidate')
        if not os.path.exists(S4_candidate_dir):
            os.makedirs(S4_candidate_dir)
        # candidate selection
        ascending_candidate, ascending_candidate_note, non_candidate_ascending_note = candidate_selection(note, ascending_pattern)
        descending_candidate, descending_candidate_note, non_candidate_descending_note = candidate_selection(note, descending_pattern)
        # save result: candidate
        np.savetxt(S4_candidate_dir+os.sep+name+'.candidate.ascending'+'.txt',ascending_candidate, fmt='%s')
        # np.savetxt('/Users/Frank/Documents/Code/Python/Test_PTNoteTransitionOverlapEval/vi_short_candidate_selection/non_candidate_ascending_note.txt',non_candidate_ascending_note)
        np.savetxt(S4_candidate_dir+os.sep+name+'.candidate.descending'+'.txt',descending_candidate, fmt='%s')
        # np.savetxt('/Users/Frank/Documents/Code/Python/Test_PTNoteTransitionOverlapEval/vi_short_candidate_selection/non_candidate_descending_note.txt',non_candidate_descending_note)


if __name__ == '__main__':
    args = parser()
    main(args)