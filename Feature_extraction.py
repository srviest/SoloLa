#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/02/27
----------------------------------------------------------------------
Feature Extactor: extract audio features of candadates segments.
----------------------------------------------------------------------
Args:
    Text files to be processed.
    Directory for storing the results.

Optional args:
    Please refer to --help.
----------------------------------------------------------------------
Returns:

"""

import glob, os, sys
import numpy as np
from scipy.io import wavfile
import math
import operator
from essentia import *
from essentia.standard import *
from GuitarTranscription_parameters import selected_features

def delta(input_sig, n=1, axis=0):
    from numpy import diff,concatenate
    if axis == 0:
        derivative = concatenate(([input_sig[0:n,:]],diff(input_sig,axis = 0)))
    elif axis == 1:
        derivative = concatenate(([input_sig[:,0:n]],diff(input_sig,axis = 1)))
    return derivative

def feature_extractor(audio, features, pool_methods=['mean', 'var', 'min', 'max', 'median', 'skew', 'kurt', 'dmean', 'dvar', 'dmean2', 'dvar2']):
    """
    Collect all files by given extension and keywords.

    :param audio:       audio signal.
    :param features:    features to be extracted.
    :returns:           feature vector.

    """
    # initiate pool
    pool = Pool()
    # extract features
    feature = LowLevelSpectralExtractor(frameSize = 2048, hopSize = 1024)(audio)
    # create feature names list
    feature_name_list = LowLevelSpectralExtractor().outputNames()
    # 'mfcc', 'barkbands', 'barkbands_kurtosis', 'barkbands_skewness', 'barkbands_spread', 
    num_feature = 0
    # loop in feature tuples to add selected features to pool
    for f in feature:
        num_feature+=1
        for ff in f:
            if feature_name_list[num_feature-1] in selected_features:
                pool.add(feature_name_list[num_feature-1], ff)
    # feature aggregation
    aggrPool = PoolAggregator(defaultStats=pool_methods)(pool)
    
    # concatenate features
    feature_vec = [] 
    featureList = aggrPool.descriptorNames()
    for name in featureList:
        feature = aggrPool[name]
        if type(feature) != float:  # for those type == array
           feature_vec = np.concatenate([feature_vec,feature], axis = 0)
        else: # for those type == float
           feature_vec.append(feature)
    # return feature_vec     
    feature_vec_all = np.concatenate((feature_vec_all,feature_vec), axis = 0)

    return feature_vec_all

def parse_input_files(args, ext='.wav'):
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
    p.add_argument('input_audios', type=str, metavar='input_audios',
                   help='audio files to be processed')
    p.add_argument('input_candidates', type=str, metavar='input_candidates',
                   help='candidate time segements of audio files to be processed')
    p.add_argument('output_dir', type=str, metavar='output_dir',
                   help='output directory.')
    p.add_argument('-m','--mfcc', nargs='*',dest = 'm', help="extrac mfcc, arguments: delta = {0,1,2}, \
                    pool_methods = {max,min,median,mean,skew,kurt,dmean,dvar,dmean2,dvar2}", default=None)
    p.add_argument('-s','--spectral', nargs='*',dest = 's', help="extrac spectral, arguments: delta = {0,1,2}, \
                    pool_methods = {max,min,median,mean,skew,kurt,dmean,dvar,dmean2,dvar2}", default=None)
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2016-03-08)')
    # parse arguments
    args = p.parse_args()
    # return args
    return args
    
def main(args):
    print 'Running feature extraction...'    
    # parse and list files to be processed
    audio_files = parse_input_files(args, ext='.wav')
        
    # create result directory
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    print '  Output directory: ', '\n', '    ', args.output_dir

    # processing
    for f in audio_files:
        # parse file name and extension
        ext = os.path.basename(f).split('.')[-1]
        name = os.path.basename(f).split('.')[0]
        # load audio into 
        audio = MonoLoader(filename = f)()
        # processing
        print '     Processing file: ', f
        candidate_type = ['ascending','descending']
        # loop in ascending and descending candidate list
        for ct in candidate_type:
            print '         EXtracting features from ', ct, ' candadites...'
            # candidate file path
            candidate_path = args.input_candidates+os.sep+name+'.'+ct+'.candidate'
            # inspect if candidate file exist and load it
            try:
                candidate = np.loadtxt(candidate_path)
            except IOError:
                print 'The candidate of ', name, ' doesn\'t exist!'
            # reshape candidate if it is in one dimension
            if candidate.shape==(2,): candidate = candidate.reshape(1,2)
            # convert seconds into samples
            candidate_sample = candidate*44100
            # create feature matrix
            feature_vec_all = np.array([])
            # loop in candidates
            for c in candidate_sample:
                # clipping audio signal
                audio_clip = audio[c[0]:c[1]]
                # extract features
                feature_vec_all = feature_extractor(audio=audio_clip, features=selected_features)
              
            # reshpe feature vector
            if feature_vec_all.size != 0:
                feature_vec_all = feature_vec_all.reshape(len(candidate_sample),len(feature_vec_all)/len(candidate_sample))
            # save result if the feature array is not empty
            if bool(feature_vec_all):
                np.savetxt(args.output_dir+os.sep+name+'.'+ct+'.candidate'+'.raw.feature', feature_vec_all, fmt='%s')


if __name__ == '__main__':
    args = parser()
    main(args)