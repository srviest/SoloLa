#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2015/10/13
--------------------------------------------------------------------------------
Script for guitar expression style recognition model training.
--------------------------------------------------------------------------------
"""


def parser():

    import argparse
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
      Main script for guitar expression style recognition model training.
      The pipeline is as follow:
      ----------------------------------------------------------------------
          S1 Melody extraction
          S2 Note tracking
          S3 Candidate selection
              S3.1 Find continuously ascending/descending (CAD) F0 sequence
                   patterns.
              S3.2 Find intersection of note and pattern (Candidate selection
                   of {bend,slide,pull-off,hammer-on,normal}).

          S4 Feature extraction
          S5 Classifier training
      ----------------------------------------------------------------------
    """)
    # general options
    p.add_argument('-me', action='store_true', default=False, 
      help='melody extraction')
    p.add_argument('-nt', action='store_true', default=False, 
      help='note tracking')
    p.add_argument('-cs', action='store_true', default=False, 
      help='candidate selection')
    p.add_argument('-fe', action='store_true', default=False, 
      help='feature extraction')
    p.add_argument('-cl', action='store_true', default=False, 
      help='classification')
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2016-04-21)')

    args = p.parse_args()
    return args


def main(args):
    from subprocess import call
    import glob, os, sys

    # input_audio = '/Users/Frank/Documents/Code/Database/test/Guitar_Licks_51_10.wav'
    # input_audio = '/Users/Frank/Documents/Code/Database/clean_tone_single_effect'
    # input_audio = '/Users/Frank/Desktop/Guitar_Score/Beatles - Let It Be Solo.wav'
    input_audio = '/Users/Frank/Documents/Code/Database/clean_tone_single'
    # output_dir = '/Users/Frank/Documents/Code/Python/GPT_experiment/Clean_Room'
    output_dir = '/Users/Frank/Documents/Code/Python/GPT_experiment/Clean_Tone'
    # output_dir = '/Users/Frank/Documents/Code/Python/GPT_experiment/All_Effects'
    # output_dir = '/Users/Frank/Documents/Code/Python/Guitar_solo_MIDI2wav'

    print '===================================================='
    print 'Training guitar expressin style recognition model...'
    print '===================================================='

    # S1.Melody extraction
    if args.me: call(['python', 'Melody_extraction.py', 
                      input_audio, 
                      output_dir+os.sep+'S1.Melody'])

    # S2.Note tracking
    if args.nt: call(['python', 'Note_tracking.py', 
                      output_dir+os.sep+'S1.Melody',
                      output_dir+os.sep+'S2.Note'])

    # S3.Candidate selection
    if args.cs: call(['python', 'Candidate_selection.py', 
                      output_dir+os.sep+'S1.Melody', 
                      output_dir+os.sep+'S2.Note', 
                      output_dir+os.sep+'S3.Candidate'])

    # S4.Feature extraction
    if args.fe: call(['python', 'Feature_extraction.py', 
                      input_audio, 
                      output_dir+os.sep+'S3.Candidate', 
                      output_dir+os.sep+'S4.Feature'])

    # S5.Classification
    # f = open(output_dir+os.sep+'S5.Classification'+os.sep+'Result.txt', 'w')
    if args.cl: call(['python', 'Classification.py', 
                      output_dir+os.sep+'S4.Feature', 
                      output_dir+os.sep+'S5.Classification',
                      'bend', 'hamm', 'slide', 'pull', 'normal', '-f', '5'])


if __name__ == '__main__':
    args = parser()
    main(args)