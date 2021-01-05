#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/02/06
----------------------------------------------------------------------
Melody Extractor: extract melody contour from audio file.
----------------------------------------------------------------------
Args:
    input_files:    Audio files to be processed. 
                    Only the wav files would be considered.
    output_dir:     Directory for storing the results.
Optional args:
    Please refer to --help.
----------------------------------------------------------------------
Returns:
    Raw melody contour:         Text file of estimated melody contour 
                                in Hz with extension of .raw.melody.
    MIDI-scale melody contour:  Text file of estimated melody contour 
                                in MIDI with extension of .MIDI.melody.
    Smoothed melody contour:    Text file of moving-averged estimated 
                                melody contour in MIDI scale with extenion 
                                of .smooth.MIDI.melody.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from past.utils import old_div
import glob, os, sys
import numpy as np
from essentia.standard import *
from guitar_trans.parameters import *

def hertz2midi(melody_contour):
    """
    Convert pitch sequence from hertz to MIDI scale.
    :param melody_contour: array of pitch sequence.
    :returns             : melody contour in MIDI scale.
    """ 
    from numpy import inf
    melody_contour_MIDI = melody_contour.copy()
    melody_contour_MIDI = np.log(old_div(melody_contour_MIDI,float(440)))
    melody_contour_MIDI =12*melody_contour_MIDI/np.log(2)+69
    melody_contour_MIDI[melody_contour_MIDI==-inf]=0

    return melody_contour_MIDI

def parse_input_files(input_files, ext='.wav'):
    """
    Collect all files by given extension and keywords.
    :param agrs:  class 'argparse.Namespace'.
    :param ext:   the string of file extension.
    :returns:     a list of stings of file name.
    """
    from os.path import basename, isdir
    import fnmatch
    import glob
    files = []

    ### check what we have (file/path)
    if isdir(input_files):
        ### use all files with ext in the given path
        files = fnmatch.filter(glob.glob(input_files+'/*'), '*'+ext)
    else:
        ### file was given, append to list
        if basename(input_files).find(ext)!=-1:
            files.append(input_files)
    print('  Input files: ')
    for f in files: print('    ', f)
    return files

def extract_melody(audio_file, save_dir=None):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    ###  initiate MELODIA
    pcm = PitchMelodia(harmonicWeight=harmonicWeight, minDuration=minDuration, 
        binResolution=binResolution, guessUnvoiced=guessUnvoiced, frameSize=frameSize, 
        hopSize=HOP_LENGTH, maxFrequency=maxFrequency, minFrequency=minFrequency, 
        filterIterations=filterIterations, magnitudeThreshold=magnitudeThreshold, 
        sampleRate=SAMPLING_RATE, peakDistributionThreshold=peakDistributionThreshold)
    audio = MonoLoader(filename = audio_file, sampleRate=SAMPLING_RATE)()
    ### run MELODIA
    melody_contour, pitchConfidence = pcm(audio)
    ### convert Hz to MIDI scale
    melody_contour_MIDI = hertz2midi(melody_contour)
    if save_dir is not None:
        ### save result: raw melody contour
        np.savetxt(save_dir+os.sep+'RawMelody.txt', melody_contour, fmt='%s')
        ### save result: MIDI-scale melody contour
        np.savetxt(save_dir+os.sep+'MidiMelody.txt', melody_contour_MIDI, fmt='%s')
    return melody_contour, melody_contour_MIDI

def main(audio_files, output_dir):
    print('============================')
    print('Running melody extraction...')
    print('============================')
    ### parse and list files to be processed
    files = parse_input_files(audio_files)
    
    ### create result directory
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print('  Output directory: ', '\n', '    ', output_dir)
    
    ### processing
    for f in files:
        name = os.path.basename(audio_file).split('.')[0]
        save_dir = os.path.join(output_dir, name)
        extract_melody(f, save_dir)
        

def parser():
    """
    Parses the command line arguments.
    :param lgd:       use local group delay weighting by default
    :param threshold: default value for threshold
    """
    import argparse
    ### define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
====================================
Script for extracting melody contour
====================================
    """)
    ### general options
    p.add_argument('input_files', type=str, metavar='input_files',
                   help='files to be processed')
    p.add_argument('output_dir', type=str, metavar='output_dir',
                   help='output directory.')
    ### version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 2.01 (2017-06-30)')
    ### parse arguments and return them
    return p.parse_args()
    
if __name__ == '__main__':
    args = parser()
    main(args.audio_files, args.output_dir)