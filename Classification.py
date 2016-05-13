#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/03/08

-------------------------------------------------------------------------------
Classification
-------------------------------------------------------------------------------
Args:
    input_files:    Audio files to be processed. 
                    Only the wav files would be considered.
    output_dir:     Directory for storing the results.

Optional args:
    Please refer to --help.
-------------------------------------------------------------------------------
Returns:
    Result:         Text file of estimated melody contour 

"""

import glob, os, sys
# sys.path.append('/Users/Frank/Documents/Code/Python')
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
    if 'hamm' in technique_type and 'pull' not in technique_type:
        feature_file = glob.glob(feature_dir+os.sep+'*.ascending.candidate.raw.feature')
    elif 'pull' in technique_type and 'hamm' not in technique_type:
        feature_file = glob.glob(feature_dir+os.sep+'*.descending.candidate.raw.feature')
    else:
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

def data_preprocessing(raw_data, data_preprocessing_method=data_preprocessing_method, scaler_path=None, output_path=None):
    from sklearn.preprocessing import Imputer, scale, robust_scale, StandardScaler

    # replace nan feature with the median of column values
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    raw_data = imp.fit_transform(raw_data)

    # remove inf and -inf
    if np.where(np.isinf(raw_data)==True)[0].size!=0:
        print 'Removing Inf and -Inf values...'
        med = np.median(raw_data, axis=0)
        axis_0 = np.where(np.isinf(raw_data)==True)[0]
        axis_1 = np.where(np.isinf(raw_data)==True)[1]
        for index in range(len(axis_0)):
            raw_data[axis_0[index], axis_1[index]]=med[axis_1[index]]

    # standardization
    if 'z-score' in data_preprocessing_method:
        print '    Standardizing data by z-score...'
        # z-score standardization
        data = scale(raw_data)
    elif 'robust z-score' in data_preprocessing_method:
        print '    Standardizing data by robust z-score...'
        # robust z-score standardization
        data = robust_scale(raw_data)

    elif 'StandardScaler' in data_preprocessing_method:
        if scaler_path=None and output_path!=None:
            print '    Standardizing data by StandardScaler method...'
            scaler = StandardScaler().fit(raw_data)
            # save scaler
            np.save(output_path+'.scaler', scaler)
            data = scaler.transform(raw_data)
        elif scaler_path!=None and output_path==None:
            print '    Standardizing data by pre-computed scaler...'
            # load scaler
            scaler = np.load(scaler_path).itme()
            data = scaler.transform(raw_data)
        elif scaler_path==None and output_path==None:
            print 'Please specify the scaler path or path to restore the scaler.'

    return data

def data_loader(technique_file_dict):
    """
    Read raw featrues from S5_feature folder and return labels y
    and data instances x with the format of numpy array.
    """
    from numpy import loadtxt, empty, asarray, hstack, vstack, savetxt
    import glob, os, sys, collections
    index_of_class = 0
    label = []
    # calculate the dimension of feature space
    f_dimension = loadtxt(technique_file_dict[technique_file_dict.keys()[0]][0]).shape[1]
    # create empty feature array for collecting feature intances
    training_instances = empty((0,f_dimension),dtype=float)

    class_data_num_str = str()
    class_data_num_dict = dict()

    # concatenate all features and make label
    for t in technique_file_dict.keys():
        num_of_instances = 0
        for f in technique_file_dict[t]:
            feature = loadtxt(f)
            try:
                feature.shape[1]
            except IndexError:
                feature = feature.reshape(1, feature.shape[0])        
            num_of_instances+=feature.shape[0]
            training_instances = vstack((training_instances,feature))
            
        label+=[index_of_class]*num_of_instances
        
        index_of_class += 1
        class_data_num_dict[t] = num_of_instances
        if t!=technique_file_dict.keys()[-1]:
            class_data_num_str += t+'_'+str(num_of_instances)+'_'
        else:
            class_data_num_str += t+'_'+str(num_of_instances)
    class_data_num_dict = collections.OrderedDict(sorted(class_data_num_dict.items()))
    # convert label list to numpy array
    
    label = asarray(label)
    label = label.reshape(label.shape[0])
 
    return label, training_instances, class_data_num_str, class_data_num_dict, f_dimension


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
    p.add_argument('-GridSearchCV', dest='GridSearchCV', default=False, action='store_true',
                    help='Exhaustive search over specified parameter values for an estimator.')
    p.add_argument('-TrainingAll', dest='TrainingAll', default=False, action='store_true',
                    help='Training with all data.')
    p.add_argument('-C','--C', type=float, dest='C',action='store',  help="penalty parmeter.", default=1)
    p.add_argument('-gamma','--gamma', type=float, dest='gamma',action='store',  help="gamma for RBF kernel SVM.", default=None)
    p.add_argument('-f','--fold', type=int, dest='f',action='store',  help="the number of fold in which data to be partitioned.", default=5)
    p.add_argument('-i','--iteration', type=int, dest='i',action='store',  help="the number of iteration of randomly partitioning data.", default=1)
    p.add_argument('-v','--validation', nargs=2, dest='v', help="cross validation. V1: number of iteration, V2: number of folds", default=None)
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2016-04-25)')
    # parse arguments
    args = p.parse_args()
    
    return args

def main(args):
    print '========================='
    print 'Running classification...' 
    print '========================='   
    # create result directory
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    print 'Output directory: ', '\n', '  ', args.output_dir
    print '\n'


    # inspect the input data classes
    technique_file_dict = collect_same_technique_feature_files(args.input_features, technique_type = args.classes)
    
    # data loader
    label, raw_data, class_data_num_str, class_data_num_dict, f_dimension = data_loader(technique_file_dict)

    # pre-processing data
    data = data_preprocessing(raw_data, data_preprocessing_method=data_preprocessing_method, output_path=args.output_dir+os.sep+class_data_num_str)
    X, y = data, label
    
    if args.GridSearchCV:
        # inspect if there exists the cross validation indices
        try:
            CVfold = np.load(args.output_dir+os.sep+class_data_num_str+'.iter'+str(args.i)+'.fold'+str(args.f)+'.CVFold.npy').item()
            print 'Load pre-partitioned cross validation folds...'
        except IOError:
            print 'Shuffling the samples and dividing them into ', args.f, ' folds...'
            CVfold = StratifiedKFold(label, args.f, shuffle=True)
            np.save(args.output_dir+os.sep+class_data_num_str+'.iter'+str(args.i)+'.fold'+str(args.f)+'.CVFold.npy', CVfold)


        # Set the parameters by cross-validation
        C_range = np.ndarray.tolist(np.logspace(-3, 4, 7, base=2))
        g_range = np.ndarray.tolist(np.logspace(-8, -3, 5, base=2))

        tuned_parameters = [{'kernel': ['rbf'], 'gamma': g_range,'C': C_range}, 
                              {'kernel': ['linear'], 'C': C_range}  ]
                            # {'kernel': ['linear'], 'C': C_range}]

        # tuned_parameters = [{'kernel': ['linear'], 'C': C_range }]

        # set the scoring metric for parameters estimation
        metrics = ['f1', 'precision']
        # {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
        # save result to file
        sys.stdout = open(args.output_dir+os.sep+'model.report', 'w')
        print '============================================================' 
        print 'Parameters and setting'
        print '============================================================' 
        print 'Targets: '
        for c_index, c in enumerate(class_data_num_dict):
            print '    %s: %s (%s)' % (c_index, c, class_data_num_dict[c])
        print 'Dimensions of feature vector: %s' % (f_dimension)
        print 'Data preprocessing method:'
        for dpm in data_preprocessing_method:
            print '    %s' % (dpm)

        # train, test and evaluation
        fold=1
        for train_index, test_index in CVfold:
            print '============================================================' 
            print 'Classification on fold %s...' % fold
            print '============================================================' 
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            for m in metrics:
                print("# Tuning hyper-parameters for %s" % m)
                # print '\n'

                clf = GridSearchCV(SVC(class_weight='balanced'), tuned_parameters, cv=4,
                                   scoring='%s_weighted' % m)
        
                clf.fit(X_train, y_train)

                print("Best parameters set found on development set:")
                # print '\n'
                print(clf.best_params_)
                # C_final+= clf.best_params_['C']
                # gamma_final+= clf.best_params_['gamma']
                print '------------------------------------------------------------' 
                print("Grid scores on development set:")
                # print '\n'
                for params, mean_score, scores in clf.grid_scores_:
                    print("%0.3f (+/-%0.03f) for %r"
                          % (mean_score, scores.std() * 2, params))
                # print '\n'
                print '------------------------------------------------------------' 
                print("Detailed classification report:")
                # print '\n'
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set.")
                # print '\n'
                y_true, y_pred = y_test, clf.predict(X_test)
                print(classification_report(y_true, y_pred))
                # print '\n'
                # save model
                np.save(args.output_dir+os.sep+class_data_num_str+'.iter'+str(args.i)+'.fold'+str(fold)+'.metric.'+m+'.model', clf)
                if clf.best_params_['kernel']=='linear':
                    clf_all = SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'], class_weight='balanced')
                elif clf.best_params_['kernel']=='rbf':
                    clf_all = SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'], gamma=clf.best_params_['gamma'], class_weight='balanced')
                clf_all.fit(X,y)
                np.save(args.output_dir+os.sep+class_data_num_str+'.iter'+str(args.i)+'.fold'+str(fold)+'.all.metric.'+m+'.model', clf_all)
            fold+=1
        print '\n'
     

    if args.TrainingAll:
        # calculate sample weight and class weight
        wt=[]
        num_list = []
        weight_list = []
        class_weight={}
        for w in range(max(y)+1):
            num=np.where(y==w)[0].size
            num_list.append(num)
            weight_list.append(1/float(num))
            wt = wt+[1/float(num)]*num
        min_num = min(num_list)
        for index, w in enumerate(weight_list):
            class_weight[index]=w/(1/float(min_num))
        print 'sample_weight: ', wt
        # training
        clf_final = SVC(C=C_final , gamma=gamma_final, class_weight='balanced')
        clf_final.fit(X, y)
        # save model
        np.save(args.output_dir+os.sep+class_data_num_str+'.iter'+str(args.i)+'.model', clf_final)

if __name__ == '__main__':
    args = parser()
    main(args)