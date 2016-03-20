#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/03/15
----------------------------------------------------------------------
Fingering arrangement: automatically arrange the guitar fingering.
----------------------------------------------------------------------
Args:
    input_files:    files to be processed. 
                    Only the .expression_style_note files would be considered.
    output_dir:     Directory for storing the results.

Optional args:
    Please refer to --help.
----------------------------------------------------------------------
Returns:
    Raw melody contour:         Text file of estimated melody contour 
                                in Hz with extenion of .raw.melody.

"""
import glob, os, sys
import numpy as np
import math
from scipy.io import wavfile


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

    """)
    # general options
    p.add_argument('input_files', type=str, metavar='input_files',
                   help='files to be processed')
    p.add_argument('output_dir', type=str, metavar='output_dir',
                   help='output directory.')
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2016-03-13)')
    # parse arguments
    args = p.parse_args()

    # return args
    return args
    

def main(args):
    print 'Running melody extraction...'
    
    # parse and list files to be processed
    files = parse_input_files(args.input_files, ext='.expression_style_note')
    
    # create result directory
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    print '  Output directory: ', '\n', '    ', args.output_dir

    # processing
    for f in files:
        # parse file name and extension
        ext = os.path.basename(f).split('.')[-1]
        name = os.path.basename(f).split('.')[0]


        
if __name__ == '__main__':
    args = parser()
    main(args)