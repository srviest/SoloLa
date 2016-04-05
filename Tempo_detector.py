#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/04/05
----------------------------------------------------------------------
Tempp detector: detect the tempo of given audio.
----------------------------------------------------------------------
Args:
    input_files:            Text files of pitch series to be processed.
    output_dir:             Directory for storing the results.
    tempo_detector_path:    The path of TempoDetector executable.

Optional args:
    Please refer to --help.
----------------------------------------------------------------------
Returns:
    BPM:          Text file of estimated bpm in order.

"""
import glob, os, sys
import subprocess as subp
import numpy as np


def parse_input_files(input_files, ext):
    """
    Collect all files by given extension.

    :param input_files:  list of input files or directories.
    :param ext:          the string of file extension.
    :returns:            a list of stings of file name.
    
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

    The TempoDetector program detects the dominant tempi according to the
    algorithm described in:
    "Accurate Tempo Estimation based on Recurrent Neural Networks and
     Resonating Comb Filters"
    Sebastian Böck, Florian Krebs and Gerhard Widmer.
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.
    This program can be run in 'single' file mode to process a single audio
    file and write the detected tempi to STDOUT or the given output file.
    $ TempoDetector single INFILE [-o OUTFILE]
    If multiple audio files should be processed, the program can also be run
    in 'batch' mode to save the detected tempi to files with the given suffix.
    $ TempoDetector batch [-o OUTPUT_DIR] [-s OUTPUT_SUFFIX] LIST OF FILES
    If no output directory is given, the program writes the files with the
    detected tempi to same location as the audio files.
    The 'pickle' mode can be used to store the used parameters to be able to
    exactly reproduce experiments.

    """)
    # general options
    p.add_argument('input_files', type=str, metavar='input_files',
                   help='files to be processed')
    p.add_argument('output_dir', type=str, metavar='output_dir',
                   help='output directory.')
    p.add_argument('-tdp',   '--TempoDetectorPath', type=str, dest='tdp',
                   help="the path of TempoDetector executable.", default='./TempoDetector')
    
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2016-04-05)')
    # parse arguments
    args = p.parse_args()

    # return args
    return args
    

def main(args):
    print 'Running tempo detector...'
    
    # parse and list files to be processed
    files = parse_input_files(args.input_files, ext='.wav')
    
    # create result directory
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    print '  Output directory: ', '\n', '    ', args.output_dir

    # processing
    for f in files:
        # parse file name and extension
        ext = os.path.basename(f).split('.')[-1]
        name = os.path.basename(f).split('.')[0]

        
        # NoteRecognizer_path = '/Users/Frank/Documents/Code/C++/Note_recognizer/MOLODIA_HMM/NoteRecognizer'
        command = [args.tdp, 'single', f]
        pipe = subp.Popen(command, stdout=subp.PIPE, startupinfo=None)
        bpm_string = pipe.stdout.read()
        bpm = []
        for line in bpm_string.splitlines():
          bpm.append(np.fromstring(line, dtype="float32", sep=' '))
        bpm = np.asarray(bpm)
        # save result: bpm
        np.savetxt(args.output_dir+os.sep+name+'.bpm', bpm, fmt='%s')
        

if __name__ == '__main__':
    args = parser()
    main(args)