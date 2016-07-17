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
                                in Hz with extenion of .raw.melody.
    MIDI-scale melody contour:  Text file of estimated melody contour 
                                in MIDI with extenion of .MIDI.melody.
    Smoothed melody contour:    Text file of moving-averged estimated 
                                melody contour in MIDI scale with extenion 
                                of .smooth.MIDI.melody.

"""
import glob, os, sys
import numpy as np
from essentia.standard import *
from GuitarTranscription_parameters import *

def mean_filter(data, kernel_size=9):
    """
    Smooth the melody contour with moving-average filter.
    :param data:            the input one-demensional to be processed.
    :param kernel_size:     the kernel size of the moving-average filter.
    :returns:               processeed data.
    """
    pseudo_data = data.copy()
    smooth = np.convolve(pseudo_data, np.ones(kernel_size)/kernel_size, mode='same')
    return smooth


def hertz2midi(melody_contour):
    """
    Convert pitch sequence from hertz to MIDI scale.

    :param melody_contour: array of pitch sequence.
    :returns             : melody contour in MIDI scale.

    """ 
    from numpy import inf
    melody_contour_MIDI = melody_contour.copy()
    melody_contour_MIDI = np.log(melody_contour_MIDI/float(440))
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

    # check what we have (file/path)
    if isdir(input_files):
        # use all files with .raw.melody in the given path
        files = fnmatch.filter(glob.glob(input_files+'/*'), '*'+ext)
    else:
        # file was given, append to list
        if basename(input_files).find(ext)!=-1:
            files.append(input_files)
    print '  Input files: '
    for f in files: print '    ', f
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
    p.add_argument('input_files', type=str, metavar='input_files',
                   help='files to be processed')
    p.add_argument('output_dir', type=str, metavar='output_dir',
                   help='output directory.')
    p.add_argument('-fs',   '--frameSize', type=int, dest='fs',  help="the frame size for computing pitch saliecnce",            default=2048)
    p.add_argument('-hs',   '--hopSize',    type=int, dest='hs',  help="the hop size with which the pitch salience function was computed.",    default=256)
    p.add_argument('-sr',   '--sampleRate', type=int, dest='sr',  help="the sampling rate of the audio signal [Hz].",              default=44100)
    p.add_argument('-maxf0','--maxf0',      type=int, dest='maxf0',   help="the maximum allowed frequency for salience function peaks (ignore contours with peaks above) [Hz].",     default=20000)

    p.add_argument('-fi','--filterIteration',      type=int, dest='fi',   help="number of iterations for the octave errors / pitch outlier filtering process",     default=2)

    p.add_argument('-minf0','--minf0',      type=int, dest='minf0',   help="the minimum allowed frequency for salience function peaks (ignore contours with peaks above) [Hz].",     default=82)
    p.add_argument('-ks','--kernelSize',    type=int, dest='ks',   help="the kernel size of median filter for smoothing the estimtated melody contour.",     default=5)
    p.add_argument('-gu', '--guessUnvoiced', action = 'store_true', dest = 'gu', help="estimate pitch for non-voiced segments by using non-salient contours when no salient ones are present in a frame.", default=True)
    p.add_argument('-no-gu', '--no-guessUnvoiced', action = 'store_false', dest = 'gu', help="turn off the guessUnvoiced.")
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2016-03-13)')
    # parse arguments
    args = p.parse_args()

    # return args
    return args
    

def main(args):
    print '============================'
    print 'Running melody extraction...'
    print '============================'
    # parse and list files to be processed
    files = parse_input_files(args.input_files)
    
    # create result directory
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    print '  Output directory: ', '\n', '    ', args.output_dir

    # processing
    for f in files:
        # parse file name and extension
        ext = os.path.basename(f).split('.')[-1]
        name = os.path.basename(f).split('.')[0]

        # S1.1 initiate MELODIA
        pcm = PitchMelodia(harmonicWeight=harmonicWeight, minDuration=minDuration, 
            binResolution=binResolution, guessUnvoiced=args.gu, frameSize=args.fs, 
            hopSize=contour_hop, maxFrequency=args.maxf0, minFrequency=args.minf0, 
            filterIterations=filterIterations, magnitudeThreshold=magnitudeThreshold, 
            sampleRate=contour_sr, peakDistributionThreshold=peakDistributionThreshold)
        audio = MonoLoader(filename = f)()
        # run MELODIA
        melody_contour, pitchConfidence = pcm(audio)
        # save result: raw melody contour
        np.savetxt(args.output_dir+os.sep+name+'.raw.melody',melody_contour, fmt='%s')
        # save result: raw melody contour
        np.savetxt(args.output_dir+os.sep+name+'.pitch_confidence',pitchConfidence, fmt='%s')
        # convert Hz to MIDI scale
        melody_contour = hertz2midi(melody_contour)
        # save result: MIDI-scale melody contour
        np.savetxt(args.output_dir+os.sep+name+'.MIDI.melody',melody_contour, fmt='%s')
        # moving averaging filtering
        melody_contour = mean_filter(melody_contour,kernel_size=mean_filter_size)
        # save result: MIDI-scaled smoothed melody contour
        np.savetxt(args.output_dir+os.sep+name+'.MIDI.smooth.melody',melody_contour, fmt='%s')

if __name__ == '__main__':
    args = parser()
    main(args)