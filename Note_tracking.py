#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/02/07
----------------------------------------------------------------------
Note Tracker: recognize note events from the estimated melody contour.
----------------------------------------------------------------------
Args:
    input_files:            Text files of pitch series to be processed.
    output_dir:             Directory for storing the results.
    note_recognizer_path:   the path of c++ note recognizer programme.

Optional args:
    Please refer to --help.
----------------------------------------------------------------------
Returns:
    Note:          Text file of estimated note events, with extenion of .note.
    Pruned note:   Text file of estimated note events which are 
                   preprocessed by prunnung the note whose duration 
                   shorter than a threshold, with extenion of .note.

"""
import glob, os, sys
import subprocess as subp
import numpy as np
import math

def note_pruning(note_pseudo, threshold=0.1):
    """
    prune the note whose length smaller than the threshold.

    :param threshold: the minimal duration of note 

    """
    note = note_pseudo.copy()
    pruned_notes = np.empty([0,3])
    for n in range(note.shape[0]):
        if note[n,2]>threshold:
            pruned_notes = np.append(pruned_notes,[note[n,:]],axis=0)
    return pruned_notes


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
    p.add_argument('-nrp',   '--NoteRecognizerPath', type=str, dest='nrp',
                   help="the path of c++ based Note tracker.", default='./NoteRecognizer')
    p.add_argument('-p',   '--prunning_note', dest='p',  help="the minimum duration of note event.",  default=0.1)
    p.add_argument('-eval', '--evaluation', type=str, default=None, dest='evaluation',
        help='Conduct evaluation. The followed argument is parent directory of annotation.')
    p.add_argument('-onset_tol', '--onset_tolerance_window', type=float, dest='onset_tol', default=0.05,
        help='Window lenght of onset tolerance. (default: %(default)s)')
    p.add_argument('-offset_rat', '--offset_tolerance_ratio', type=float, dest='offset_rat', default=20,
        help='Window lenght of onset tolerance. (default: %(default)s)')
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2016-03-07)')
    # parse arguments
    args = p.parse_args()

    # return args
    return args
    

def main(args):
    print '========================'
    print 'Running note tracking...'
    print '========================'
    # parse and list files to be processed
    files = parse_input_files(args.input_files, ext='.raw.melody')
    
    # create result directory
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    print '  Output directory: ', '\n', '    ', args.output_dir

    # processing
    for f in files:
        # parse file name and extension
        ext = os.path.basename(f).split('.')[-1]
        name = os.path.basename(f).split('.')[0]

        # S2.1 recognize note event in estimated melody contour
        # NoteRecognizer_path = '/Users/Frank/Documents/Code/C++/Note_recognizer/MOLODIA_HMM/NoteRecognizer'
        command = [args.nrp, f]
        pipe = subp.Popen(command, stdout=subp.PIPE, startupinfo=None)
        note_string = pipe.stdout.read()
        note = []
        for line in note_string.splitlines():
          note.append(np.fromstring(line, dtype="float32", sep=' '))
        # convert list into ndarray
        note = np.asarray(note)
        # save result: note event
        np.savetxt(args.output_dir+os.sep+name+'.raw.note',note, fmt='%s')
        # S2.2 note pruning
        pruned_note = note_pruning(note, threshold=args.p)
        # save result: prunied note event
        np.savetxt(args.output_dir+os.sep+name+'.pruned.note',pruned_note, fmt='%s')

        if args.evaluation:
            from GuitarTranscription_evaluation import note_evaluation
            print '  Evaluating...'            
            annotation = np.loadtxt(args.evaluation+os.sep+name+'.note.answer')
            note_evaluation(annotation, note, pruned_note, args.output_dir, 
                name, onset_tolerance=args.onset_tol, offset_ratio=args.offset_rat)
            
        

if __name__ == '__main__':
    args = parser()
    main(args)