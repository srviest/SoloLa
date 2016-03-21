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
sys.path.append('/Users/Frank/Documents/Code/Python')
sys.path.append('/Users/Frank/Documents/Code/Python/libsvm-3.18/python')
sys.path.append('/Users/Frank/Documents/Code/Python/libsvm-3.18/tools')
from grid import *
import numpy as np
from scipy.io import wavfile
import math
import subprocess as subp
from scipy.io import wavfile
from svmutil import *
from PTNoteTransitionOverlapEval import *
from io_tool import audio2wave


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

def nparray2saprse_array(feature_matrix):
    """
    Transform numpy array into LIBSVM-format sparse matrix (only record non-zero entries).

    Input:   nparray                numpy array of feature instances.

    Output:  sparse_matrix          A list of dictionaries.
                                    Example: [{'1': v1, '2': v2 , '4': v4},
                                              {'1': v1, '2': v2 , '3': v3},
                                              {'1': v1, '3': v3 , '4': v4},
                                              {'1': v1, '4': v4}]
    """
    sparse_matrix = []
    for row in feature_matrix:
        col=1
        xi = dict()
        for e in row:
            if e!=0: xi[int(col)] = float(e)
            col+=1 
        sparse_matrix+=[xi]
    return sparse_matrix
    
    """""""""""""""""""""""""""""""""""""""""""""
    sparse_matrix = []
    for row in feature_matrix:
        col=1
        xi = []
        for e in row:
            if e!=0: xi.append(str(col)+':'+str(e))
            col+=1 
        sparse_matrix+=[xi]
    return sparse_matrix
    """""""""""""""""""""""""""""""""""""""""""""
def transform_2_libsvm_format(data_nparray, output_file):
    """
    Write nparray-format data into LIBSVM-formated files (only record non-zero entries).

    Input:   data_nparray               numpy array of feature instances.
                                        First column: the class label of the data.
                                        Remained columns: feature instances.
             output_file                the path of output file.

    """
    from os.path import dirname, exists
    from os import makedirs
    pdir = dirname(output_file)
    if not exists(pdir): makedirs(pdir)

    libsvm_data = open(output_file, 'wb')
    for row in data_nparray:
        col=0 
        for e in row:
            if col==0:
                libsvm_data.write(str(e))
                libsvm_data.write(' ')
            if e!=0 and col!=0: 
                libsvm_data.write(str(col))
                libsvm_data.write(':')
                libsvm_data.write(str(e))
                libsvm_data.write(' ')
            col+=1 
        libsvm_data.write('\n')
    libsvm_data.close()

def data_preprocessing(raw_data):
    from sklearn.preprocessing import scale
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
    from sklearn.preprocessing import Imputer
    # replace nan feature with the median of column values
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)

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
            # replace nan feature with the median of column values
            feature = imp.fit_transform(feature)
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

def k_fold_data_partitioner(label, output_dir, class_name_list, iteration=1, fold=4):
    """
    Partition label of training instanecs by k fold method to execute cross validation.

    Input:  label                   the list of labels.
                                    Example: [1,1,1,1,1,1,1,2,2,2,2,2,3,3,3]

            class_name_list         list of string of class names.
                                    Example: [bend, pull, slide]

            output_dir              the path of output directory to store the result.
            iteration               the times of which leave-one-out data partition to be executed.
            fold                    the number of fold in which data to be partitioned.
    """
    import glob, os, sys
    from numpy import save, load, ndarray
    import random
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # check the type of label
    if type(label)!=list:
        label = ndarray.tolist(label)

    num_of_class = int(label[-1])
    list_of_class = range(1,num_of_class+1)
    list_of_fold = range(1,fold+1)
    list_of_iter = range(1,iteration+1)
    
    print 'Amount of data in each class:'
    for n in list_of_class: print 'Class %s: '%n, label.count(n)
    # loop in iteration
    for i in list_of_iter:
        # create empty dictionary for storing random index
        CVfold = dict()
        total_num_data = 0
        # create empty string of file name
        name = str()
        # initiate empty dictionary to store the indices
        for f in list_of_fold:
            CVfold[str(f)] = dict()
            CVfold[str(f)]['train'] = []
            CVfold[str(f)]['test'] = []
        for n in list_of_class:
            num_data_n_class = label.count(n)
            random_indices_list = range(total_num_data, total_num_data+num_data_n_class)
            random.shuffle(random_indices_list)
            num_data_n_class_per_fold = num_data_n_class/fold
            total_num_data_n_class = 0
            total_num_data+=num_data_n_class
            # name the file
            if n!=list_of_class[-1]:
                name+=class_name_list[n-1]+'_'+str(num_data_n_class)+'_'
            else:
                name+=class_name_list[n-1]+'_'+str(num_data_n_class)
            # loop in fold to separate random indices
            for f in list_of_fold:
                # create testing indices
                if f!=list_of_fold[-1]:
                    test_indices_list = random_indices_list[total_num_data_n_class:total_num_data_n_class+num_data_n_class_per_fold]
                else:
                    test_indices_list = random_indices_list[total_num_data_n_class::]
                # create training indices
                train_indices_list = list(set(random_indices_list) - set(test_indices_list))
                CVfold[str(f)]['test']+=test_indices_list
                CVfold[str(f)]['train']+=train_indices_list
                total_num_data_n_class+=num_data_n_class_per_fold
        # sort the indices order
        for f in list_of_fold:
            CVfold[str(f)]['test'].sort()
            CVfold[str(f)]['train'].sort()
        # save result
        save(output_dir+os.sep+name+'.iter'+str(i)+'.fold'+str(fold)+'.npy', CVfold)

def evaluation(g_label, p_label):
    """
    Evaluate recall, precision and f-score.

    Input:  g_label                 the list of ground labels.
                                    Example: [1,1,1,1,1,1,1,2,2,2,2,2,3,3,3]
            p_label                 the list of predicted labels.
    """
    import glob, os, sys
    from numpy import save, load, asarray, where, copy, delete
    import random
    # if not os.path.exists(output_dir): os.makedirs(output_dir)

    if type(g_label)==list:
        g_label=asarray(g_label)
    if type(p_label)==list:
        p_label=asarray(p_label)
    
    num_of_class = int(g_label[-1])
    list_of_class = range(1,num_of_class+1)

    result=dict()
    sum_fscore=0
    for c in list_of_class:
        g_s = where(g_label==c)[0][0]
        g_e = where(g_label==c)[0][-1]
        pseuso_p_label = copy(p_label)
        local_p = p_label[g_s:g_e+1]
        p_i = where(p_label==c)[0]
        TP = list(p_label[g_s:g_e+1]).count(c)
        FP = list(delete(pseuso_p_label, range(g_s,g_e+1))).count(c)
        FN = (g_e-g_s+1)-TP
        Recall = TP/float(TP+FN)
        Precision = TP/float(TP+FP)
        Fscore = 2*(Recall*Precision)/float(Recall+Precision)
        result[str(c)] = dict()
        result[str(c)]['TP'] = TP
        result[str(c)]['FP'] = FP
        result[str(c)]['FN'] = FN
        result[str(c)]['Recall'] = Recall
        result[str(c)]['Precision'] = Precision
        result[str(c)]['F-Score'] = Fscore
        sum_fscore+=Fscore

    mean_fscore = sum_fscore/float(len(list_of_class))

    return result, mean_fscor


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

    from sklearn.cross_validation import StratifiedKFold
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC
    # from __future__ import print_function

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
        fold+=1
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


if __name__ == '__main__':
    args = parser()
    main(args)