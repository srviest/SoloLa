#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/03/08

----------------------------------------------------------------------
Classification
----------------------------------------------------------------------
Args:
    input_files:    Audio files to be processed. 
                    Only the wav files would be considered.
    output_dir:     Directory for storing the results.

Optional args:
    Please refer to --help.
----------------------------------------------------------------------
Returns:
    Result:         Text file of estimated melody contour 

"""

import glob, os, sys
# sys.path.append('/Users/Frank/Documents/Code/Python')
# sys.path.append('/Users/Frank/Documents/Code/Python/libsvm-3.18/python')
# sys.path.append('/Users/Frank/Documents/Code/Python/libsvm-3.18/tools')
# from grid import *
import numpy as np
import math
import subprocess as subp
# from svmutil import *
from GuitarTranscription_parameters import data_preprocessing_method
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def collect_same_technique_feature_files(feature_dir, technique_type = ['bend', 'pull', 'normal','hamm', 'slide']):
    """
    Collect feature files of same technique into a dictionary.

    Input:  output_dir                          a string of directory of extracted featrue files.
            technique_type                      a list of string containing guitar technique.

    Output: technique_file_dict                 a dictionary.
                                                key: type of technique, value: extacted feature files.
                                                Example: technique_file_dict = {'bend':[file1, file2]
                                                                                'pull':[file3, file4
                                                                                'hamm':[file5, file6}
    """
    import glob, os, sys, collections
    # inspect 
    feature_file = glob.glob(feature_dir+os.sep+'*.raw.feature')
    feature_file.sort()
    technique_type.sort()
    technique_file_dict = dict()
    for t in technique_type:
        technique_file_dict[t] = []
        for f in feature_file:
            if os.path.basename(f).find(t)!=-1:
                technique_file_dict[t].append(f)
    technique_file_dict = collections.OrderedDict(sorted(technique_file_dict.items()))
    return technique_file_dict

def data_preprocessing(raw_data):
    from sklearn.preprocessing import Imputer, scale

    # replace nan feature with the median of column values
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    raw_data = imp.fit_transform(raw_data)

    # z-score standardization
    data = scale(raw_data)

    return data

def data_loader(technique_file_dict):
    """
    Read raw featrues from S5_feature folder and return labels y
    and data instances x with the format of numpy array.
    """
    from numpy import loadtxt, empty, asarray, hstack, vstack, savetxt
    import glob, os, sys
    index_of_class = 0
    label = []
    # calculate the dimension of feature space
    f_dimension = loadtxt(technique_file_dict[technique_file_dict.keys()[0]][0]).shape[1]
    print 'The dimendion of features is: ', f_dimension
    # create empty feature array for collecting feature intances
    training_instances = empty((0,f_dimension),dtype=float)

    class_data_num = str()

    # concatenate all features and make label
    for t in technique_file_dict.keys():
        num_of_instances = 0
        for f in technique_file_dict[t]:
            feature = loadtxt(f)
            num_of_instances+=feature.shape[0]
            training_instances = vstack((training_instances,feature))
        label+=[index_of_class]*num_of_instances
        index_of_class += 1
        if t!=technique_file_dict.keys()[-1]:
            class_data_num += t+'_'+str(num_of_instances)+'_'
        else:
            class_data_num += t+'_'+str(num_of_instances)
    # convert label list to numpy array
    label = asarray(label)
    label = label.reshape(label.shape[0])
 
    return label, training_instances, class_data_num


def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def parser():
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
    # p.add_argument('input_files', type=str, metavar='input_files', nargs='+',
                   # help='files to be processed')    
    p.add_argument('input_features', type=str, metavar='input_features',
                   help='files to be processed')
    p.add_argument('output_dir', type=str, metavar='output_dir',
                   help='output directory.')    
    p.add_argument('classes',  nargs='+',  type=str, metavar='classes',  help="the types to which data belong")
    p.add_argument('-f','--fold', type=int, dest='f',action='store',  help="the number of fold in which data to be partitioned.", default=5)
    p.add_argument('-i','--iteration', type=int, dest='i',action='store',  help="the number of iteration of randomly partitioning data.", default=1)
    p.add_argument('-v','--validation', nargs=2, dest='v', help="cross validation. V1: number of iteration, V2: number of folds", default=None)
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2016-03-13)')
    # parse arguments
    args = p.parse_args()
    
    return args

def main(args):
    print 'Running classification...'    
    # create result directory
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    print 'Output directory: ', '\n', '  ', args.output_dir
    print '\n'
    # inspect the input data classes
    technique_file_dict = collect_same_technique_feature_files(args.input_features, technique_type = args.classes)
    print 'Targets: '
    print technique_file_dict.keys()

    # data loader
    label, raw_data, class_data_num = data_loader(technique_file_dict)
  
    # pre-processing data
    data = data_preprocessing(raw_data)
    X, y = data, label
    
    # inspect if there exists the cross validation indices
    try:
        CVfold = np.load(args.output_dir+os.sep+class_data_num+'.iter'+str(args.i)+'.fold'+str(args.f)+'.CVFold.npy').item()
        print 'Load pre-partitioned cross validation folds...'
    except IOError:
        print 'Shuffling the samples and dividing them into ', args.f, ' folds...'
        CVfold = StratifiedKFold(label, args.f, shuffle=True)
        np.save(args.output_dir+os.sep+class_data_num+'.iter'+str(args.i)+'.fold'+str(args.f)+'.CVFold.npy', CVfold)


    # Set the parameters by cross-validation
    C_range = np.ndarray.tolist(np.logspace(-3, 4, 7, base=2))
    g_range = np.ndarray.tolist(np.logspace(-8, -3, 5, base=2))

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': g_range,'C': C_range}]
                        # {'kernel': ['linear'], 'C': C_range}]

    # tuned_parameters = [{'kernel': ['linear'], 'C': C_range }]

    # set the scoring metric for parameters estimation
    metrics = ['f1']
    # save result to file
    sys.stdout = open(args.output_dir+os.sep+'Result.txt', 'w')
    # train, test and evaluation
    fold=1
    for train_index, test_index in CVfold:
        print 'Fold %s...' % fold
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        for metric in metrics:
            print("# Tuning hyper-parameters for %s" % metric)
            # print '\n'

            clf = GridSearchCV(SVC(), tuned_parameters, cv=4,
                               scoring='%s_weighted' % metric)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            # print '\n'
            print(clf.best_params_)
            # print '\n'
            print("Grid scores on development set:")
            # print '\n'
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() * 2, params))
            # print '\n'

            print("Detailed classification report:")
            # print '\n'
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            # print '\n'
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            # print '\n'
            # save model
            np.save(args.output_dir+os.sep+class_data_num+'.iter'+str(args.i)+'.fold'+str(fold)+'.model', clf)
        fold+=1


if __name__ == '__main__':
    args = parser()
    main(args)