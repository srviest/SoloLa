#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/01/25

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


def collect_same_technique_feature_files(output_dir, technique_type = ['bend', 'pull', 'normal','hamm', 'slide']):
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
    feature_dir = output_dir+os.sep+'S5_feature'
    # inspect 
    feature_file = glob.glob(feature_dir+os.sep+'*.txt')
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

def data_preprocessing(technique_file_dict):
    """
    Read raw featrues from S5_feature folder and return labels y
    and data instances x with the format of numpy array.
    """
    from numpy import loadtxt, empty, asarray, hstack, vstack, savetxt
    import glob, os, sys
    from sklearn.preprocessing import Imputer, scale
    # replace nan feature with the median of column values
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)

    num_of_class = 0
    label = []
    # calculate the dimension of feature space
    f_dimension = loadtxt(technique_file_dict[technique_file_dict.keys()[0]][0]).shape[1]
    print 'The dimendion of features is: ', f_dimension
    # create empty feature array for collecting feature intances
    training_instances = empty((0,f_dimension),dtype=float)

    # concatenate all features and make label
    for t in technique_file_dict.keys():
        num_of_class += 1
        num_of_instances = 0
        for f in technique_file_dict[t]:
            feature = loadtxt(f)
            # replace nan feature with the median of column values
            feature = imp.fit_transform(feature)
            num_of_instances+=feature.shape[0]
            training_instances = vstack((training_instances,feature))
        label+=[num_of_class]*num_of_instances

    # convert label list to numpy array
    label = asarray(label)
    label = label.reshape(label.shape[0],1)

    # z-score standardization
    training_instances = scale(training_instances)
    data = hstack((label, training_instances))
 
    return data

def leave_one_out_data_partitioner(label, output_dir, class_name_list, iteration=1, fold=4):
    """
    Partition label of training instanecs by leave-one-out method to execute cross validation.

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

"""
def write_result( output_file):

"""




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
    
    p.add_argument('classes',  nargs='+',  type=str, metavar='classes',  help="the types to which data belong")
    p.add_argument('-f','--fold', type=int, dest='f',action='store',  help="the number of fold in which data to be partitioned.", default=5)
    p.add_argument('-i','--iteration', type=int, dest='i',action='store',  help="the number of iteration of randomly partitioning data.", default=1)
    p.add_argument('-v','--validation', nargs=2, dest='v', help="cross validation. V1: number of iteration, V2: number of folds", default=None)
    
    args = p.parse_args()
    
    return args

def main(args):
    # create result directory
    cwd = os.getcwd()
    output_dir = os.path.join(cwd,'BaselinePT')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print 'Output directory: ', '\n', '  ', output_dir

    data_class_num = 'bend_178_hamm_86_normal_114_pull_76_slide_172'

    technique_file_dict = collect_same_technique_feature_files(output_dir, technique_type = args.classes)
    print technique_file_dict.keys()
    # pre-processing data
    clean_data = data_preprocessing(technique_file_dict)
    # np.savetxt(output_dir+os.sep+'S6_classifier'+os.sep+'clean_data.txt',clean_data,fmt='%s')

    # take the label column
    label = clean_data[:,0]
    
    # partition data
    if args.f:
        leave_one_out_data_partitioner(label, output_dir=output_dir+os.sep+'S6_classifier', class_name_list=args.classes, iteration=args.i, fold=args.f)

    # loop in iteration
    for i in range(1,args.i+1):
        # load data indices of each partition
        CVfold = np.load(output_dir+os.sep+'S6_classifier'+os.sep+data_class_num+'.iter'+str(i)+'.fold'+str(args.f)+'.npy').item()
        # loop in fold
        for f in range(1,args.f+1):
            train_indices_list = CVfold[str(f)]['train']
            test_indices_list = CVfold[str(f)]['test']
            train_data_nparray = clean_data[train_indices_list,:]
            test_data_nparray = clean_data[test_indices_list,:]
            train_file = output_dir+os.sep+'S6_classifier'+os.sep+data_class_num+'.iter'+str(i)+'.fold'+str(args.f)+os.sep+'train'+'.iter'+str(i)+'.fold'+str(f)+'.txt'
            # convert numpy array tranining data into libsvm-format data
            transform_2_libsvm_format(train_data_nparray, output_file=train_file)
            test_file = output_dir+os.sep+'S6_classifier'+os.sep+data_class_num+'.iter'+str(i)+'.fold'+str(args.f)+os.sep+'test'+'.iter'+str(i)+'.fold'+str(f)+'.txt'
            # convert numpy array testing data into libsvm-format data
            transform_2_libsvm_format(test_data_nparray, output_file=test_file)
            print 'Searching for the optimal parameters for SVM in iteration '+str(i)+', fold: '+str(f)+'...'
            # disabel the print
            blockPrint()
            # grid search for the optimal parameters
            acc, param = find_parameters(train_file, '-log2c -5,5,1 -log2g -8,0,1 -v 4 -gnuplot /usr/local/Cellar/gnuplot/4.6.5/bin/gnuplot')
            # restore the print
            enablePrint()
            print '    Cross-validation accurary: ', acc
            c = param['c']
            g = param['g']
            print '    Best c: ', c
            print '    Best g: ', g
            # read training data
            train_label, train_instance = svm_read_problem(train_file)
            # train
            m = svm_train(train_label, train_instance, '-c '+str(c)+' -g '+str(g))
            # read testing data
            test_label, test_instance = svm_read_problem(test_file)
            # predict
            p_label, p_acc, p_val = svm_predict(test_label, test_instance, m)
            # evaluation
            result, mean_fscore = evaluation(test_label, p_label)
            print 'Mean Fscore: ', mean_fscore
            print 'Result: ', result 
            
            # write_result(data_class_num, iter=i,fold=args.f,output_file)



if __name__ == '__main__':
    args = parser()
    main(args)