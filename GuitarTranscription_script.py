#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/03/13
----------------------------------------------------------------------
Script for transforming audio into sheet music.

"""

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
    for i in input_files:
        # check what we have (file/path)
        if isdir(i):
            # use all files with .raw.melody in the given path
            files = fnmatch.filter(glob.glob(i+'/*'), '*'+ext)
        else:
            # file was given, append to list
            if basename(i).find(ext)!=-1:
                files.append(i)
    print '  Input files: '
    for f in files: print '    ', f
    return files

def parser():

    import argparse
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description="""
    Main script for Guitar Playing Technique classification experiment.
    The pipeline is as follow.
    ------------------------------------------------------------------------
        S0. BPM estimation
        S1. Monaural source separation
        S2. Melody extraction
        S3. Note tracking
        S4. Expression style recognition
        S5. Fingering arrangement
    ------------------------------------------------------------------------
    Usage: 
        $ Python GuitarTranscrption_script.py ./Input_audio.wav ./Result

    """)
    # general options
    p.add_argument('input_files', type=str, metavar='input_files', nargs='+',
                   help='files to be processed')    
    p.add_argument('output_dir', type=str, metavar='output_dir',
                   help='output directory.')
    p.add_argument('-bpme', action='store_true', default=False, help='BPM estimation')
    p.add_argument('-mse', action='store_true', default=False, help='monaural source separation')
    p.add_argument('-me', action='store_true', default=False, help='melody extraction')
    p.add_argument('-nt', action='store_true', default=False, help='note tracking')
    p.add_argument('-esr', action='store_true', default=False, help='expression style recognition')
    p.add_argument('-fa', action='store_true', default=False, help='fingering arrangement')
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2016-04-04)')
    # parse arguments
    args = p.parse_args()
    return args


def main(args):
    from subprocess import call
    import glob, os, sys

    # input_audio = '/Users/Frank/Documents/Code/Database/test/Guitar_Licks_51_10.wav'
    # input_audio = '/Users/Frank/Documents/Code/Database/clean_tone_single_effect'
    input_audio = '/Users/Frank/Documents/Code/Database/clean_tone_single_effect'
    # output_dir = '/Users/Frank/Documents/Code/Python/GPT_experiment/Clean_Room'
    output_dir = '/Users/Frank/Documents/Code/Python/GPT_experiment/All_Effects'
    
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
        print '--------------------------------------------------------------'
        print 'Transforming the song "%s" into sheet music...' % name
        print '--------------------------------------------------------------'

        
        # S0. BPM estimation
        if args.bpme: call(['python', 'BPM_estimation.py',
                           f,
                           args.output_dir+os.sep+name+os.sep+'S0.BPM'])

        # S1. Monaural source separation
        if args.mse: call(['python', 'Monaural_source_separation.py',
                           f,
                           args.output_dir+os.sep+name+os.sep+'S1.IsolatedGuitar'])

        # S2. Melody extraction
        if args.me: call(['python', 'Melody_extraction.py',
                          args.output_dir+os.sep+name+os.sep+'S1.IsolatedGuitar',
                          args.output_dir+os.sep+name+'S2.Melody'])

        # S3. Note tracking
        if args.nt: call(['python', 'Note_tracking.py',
                          args.output_dir+os.sep+name+'S2.Melody',
                          args.output_dir+os.sep+name+'S3.Note'])

        # S4. Expression style recognition
        if args.esr: call(['python', 'Expression_style_recognition.py',
                           args.output_dir+os.sep+name+os.sep+'S1.IsolatedGuitar',
                           args.output_dir+os.sep+name+'S2.Melody',
                           args.output_dir+os.sep+name+'S3.Note',
                           args.output_dir+os.sep+name+'S4.ExpressionStyle'])

        # S5. Fingering arramgement
        if args.fa: call(['python', 'Fingering_arramgement.py', 
                          args.output_dir+os.sep+'S4.ExpressionStyle', 
                          args.output_dir+os.sep+'S5.Fingering'])
    

if __name__ == '__main__':
    args = parser()
    main(args)