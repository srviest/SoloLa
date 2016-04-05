#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2015/10/13

Guitar playing technique baseline experiment (S5).
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
from essentia import *
from essentia.standard import *
import subprocess as subp
# import matplotlib.pyplot as plt
from scipy.io import wavfile
import operator
from svmutil import *
from PTNoteTransitionOverlapEval import *
from io_tool import audio2wave
# import matplotlib.pyplot as plt
# from pylab import *
# from librosa.feature import *


def delta(input_sig, n=1, axis=0):
    from numpy import diff,concatenate
    if axis == 0:
        derivative = concatenate(([input_sig[0:n,:]],diff(input_sig,axis = 0)))
    elif axis == 1:
        derivative = concatenate(([input_sig[:,0:n]],diff(input_sig,axis = 1)))
    return derivative

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
    p.add_argument('-m','--mfcc', nargs='*',dest = 'm', help="extrac mfcc, arguments: delta = {0,1,2}, \
                    pool_methods = {max,min,median,mean,skew,kurt,dmean,dvar,dmean2,dvar2}", default=None)
    p.add_argument('-s','--spectral', nargs='*',dest = 's', help="extrac spectral, arguments: delta = {0,1,2}, \
                    pool_methods = {max,min,median,mean,skew,kurt,dmean,dvar,dmean2,dvar2}", default=None)
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2014-11-02)')
    # parse arguments
    args = p.parse_args()
    # return args
    return args    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2014-11-02)')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return args
    return args
    
def main(args):
    print 'Running Baseline Playing Technique S5.'
    audio_dir = '/Users/Frank/Documents/Code/Database/test'
    candidate_dir = '/Users/Frank/Documents/Code/Python/BaselinePT'
    # determine the files to be processed 
    files = []
    for f in args.input_files:
        # check what we have (file/path)
        if os.path.isdir(f):
            # use all files in the given path
            files = glob.glob(f + '/*.wav')           
        else:
            # file was given, append to list
            files.append(f)

    # list files to be processed 
    print 'Input files: 'for f in files: print '  ', f
        
    # create result directory
    cwd = os.getcwd()
    output_dir = os.path.join(cwd,'BaselinePT')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print 'Output directory: ', '\n', '  ', output_dir

    # processing
    for f in files:        
        # parse file name and extension
        ext = os.path.basename(f).split('.')[-1]
        file_name = os.path.basename(f).split('.')[0]
        # load audio into 
        audio = MonoLoader(filename = f)()

        """
        S5 Extract features from audio signal
        """
        # set featrues to be extracted
        selected_features = ['hfc', 'pitch', 'pitch_instantaneous_confidence', \
            'pitch_salience', 'silence_rate_20dB', 'silence_rate_30dB', 'silence_rate_60dB', \
            'spectral_complexity', 'spectral_crest', 'spectral_decrease', 'spectral_energy', \
            'spectral_energyband_low', 'spectral_energyband_middle_low', 'spectral_energyband_middle_high', \
            'spectral_energyband_high', 'spectral_flatness_db', 'spectral_flux', 'spectral_rms', \
            'spectral_rolloff', 'spectral_strongpeak', 'zerocrossingrate', 'inharmonicity', 'tristimulus', \
            'oddtoevenharmonicenergyratio']
        # create result directory
        S5_feature_dir = os.path.join(output_dir,'S5_feature')
        if not os.path.exists(S5_feature_dir): os.makedirs(S5_feature_dir)
        # processing
        print 'Processing file: ', f
        candidate_type = ['ascending','descending']
        # loop in ascending and descending candidate list
        for ct in candidate_type:
            print '  Processing', ct, '...'
            # load candidates 
            candidate = np.loadtxt(output_dir+os.sep+'S4_candidate'+os.sep+file_name+'_'+ct+'.txt')
            # convert seconds into samples
            candidate_sample = candidate*44100
            # create feature matrix
            feature_vec_all = np.array([])
            # loop in candidates
            for c in candidate_sample:
                # clipping audio signal
                audio_clip = audio[c[0]:c[1]]
                # initiate pool
                pool = Pool()
                # extract features
                feature = LowLevelSpectralExtractor(frameSize = 2048, hopSize = 1024)(audio_clip)
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
                aggrPool = PoolAggregator(defaultStats = ['mean', 'var', 'min', 'max', 'median', 'skew', 'kurt', 'dmean', 'dvar', 'dmean2', 'dvar2'])(pool)
                
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
            # reshpe feature vector
            if feature_vec_all.size != 0:
                feature_vec_all = feature_vec_all.reshape(len(candidate_sample),len(feature_vec_all)/len(candidate_sample))
            # save result
            np.savetxt(S5_feature_dir+os.sep+file_name+'_'+ct+'.txt', feature_vec_all, fmt='%s')



if __name__ == '__main__':
    args = parser()
    main(args)