#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Date: 2016/03/13
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

    import argparse
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description="""
    Main script for transforming muiscal audio recording into sheet music.
    The pipeline is as follow.
    ------------------------------------------------------------------------
        S0. Meter tracking
        S1. Monaural source separation
        S2. Melody extraction
        S3. Note tracking
        S4. Expression style recognition
        S5. Fingering arrangement
    ------------------------------------------------------------------------
    Usage: 
        $ python GuitarTranscrption_script.py ./Input_audio.wav ./Result

    """)
    # general options
    p.add_argument('input_files', type=str, metavar='input_files',
                   help='files to be processed')    
    p.add_argument('output_dir', type=str, metavar='output_dir',
                   help='output directory.')
    p.add_argument('-dbt', action='store_true', default=False, 
        help='Downbeat tracking')
    p.add_argument('-mse', action='store_true', default=False, 
        help='Monaural source separation')
    p.add_argument('-me', action='store_true', default=False, 
        help='Melody extraction')
    p.add_argument('-nt', action='store_true', default=False, 
        help='Note tracking')
    p.add_argument('-esr', action='store_true', default=False, 
        help='Expression style recognition')
    p.add_argument('-fa', action='store_true', default=False, 
        help='Fingering arrangement')
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
    
    # output_dir = '/Users/Frank/Documents/Code/Python/GPT_experiment/Clean_Room'
    # args.input_files = '/Users/Frank/Documents/Code/Database/test'
    # args.output_dir = '/Users/Frank/Documents/Code/Python/Test_160521_SoloLa!'
    
    
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

        
        # S0. Downbeat tracking
        if args.dbt: call(['python', 'Downbeat_tracking.py',
                           f,
                           args.output_dir+os.sep+name+os.sep+'S0.Beat'])

        # S1. Monaural source separation
        if args.mse: call(['python', 'Monaural_source_separation.py',
                           f,
                           args.output_dir+os.sep+name+os.sep+'S1.IsolatedGuitar'])

        # S2. Melody extraction
        if args.me: call(['python', 'Melody_extraction.py',
                          args.output_dir+os.sep+name+os.sep+'S1.IsolatedGuitar',
                          args.output_dir+os.sep+name+os.sep+'S2.Melody']) 

        # S3. Note tracking
        if args.nt: call(['python', 'Note_tracking.py',
                          args.output_dir+os.sep+name+os.sep+'S2.Melody',
                          args.output_dir+os.sep+name+os.sep+'S3.Note'])

        # S4. Expression style recognition
        if args.esr: call(['python', 'Expression_style_recognition.py',
                           args.output_dir+os.sep+name+os.sep+'S1.IsolatedGuitar',
                           args.output_dir+os.sep+name+os.sep+'S2.Melody',
                           args.output_dir+os.sep+name+os.sep+'S3.Note',
                           '/Users/Frank/Documents/Code/Python/GPT_experiment/Pre-train_model/S5.Classification_bend_pull_normal_hamm_slide/bend_1169_hamm_1169_normal_1169_pull_1169_slide_1169.iter1.fold1.all.metric.f1.model.npy',
                           args.output_dir+os.sep+name+os.sep+'S4.ExpressionStyle', 
                           '-scaler_path', 
                           '/Users/Frank/Documents/Code/Python/GPT_experiment/Pre-train_model/S5.Classification_bend_pull_normal_hamm_slide/bend_1169_hamm_1169_normal_1169_pull_1169_slide_1169.iter1.fold1.all.metric.f1', 
                           '-debug'])

        # S5. Fingering arramgement
        if args.fa: call(['python', 'Fingering_arrangement.py', 
                          args.output_dir+os.sep+name+os.sep+'S4.ExpressionStyle', 
                          args.output_dir+os.sep+name+os.sep+'S5.Fingering'])
    

if __name__ == '__main__':
    args = parser()
    main(args)