#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/04/24
--------------------------------------------------------------------------------
Script for evaluating performaces of guitar transcription
--------------------------------------------------------------------------------
expression_style_note:  Text file of array, storing the onset, offset 
                            and pitch of each note as well as its expression.
                            The file is attached with .expression_style_note
                             extenion.

    Example:
        (0)    (1)   (2)   (3)   (4)   (5)   (6)   (7)   (8)   (9)  (10)  (11)
        Pit     On   Dur  PreB     B     R     P     H     S    SI    SO     V    
    [    66   1.24   0.5     2     0     0     0     0     1     2     1     1]

    Pi:     pitch (MIDI number)
    On:     onset (sec.)
    Dur:    duration (sec.)

    PreB:   pre-bend


    B:      string bend (0 for none,
                         1 for bend by 1 semitone,
                         2 for bend by 2 semitone,
                         3 for bend by 3 semitone, 
                         
    R:      release  (0: none, 
                      1: release by 1 semitone,
                      2: release by 2 semitone,
                      3: release by 3 semitone)

    P:      pull-off (0: none, 
                      1: pull-off start,
                      2: pull-off stop)

    H:      hammer-on (0: none,
                       1: hammer-on start,
                       2: hammer-on stop)

    S:      legato slide (0: none,
                          1: legato slide start, 
                          2: legato slide stop, 
                
    SI:     slide in (0: none,
                      1: slide in from below,
                      2: slide in from above)

    SO:     slide out (0: none,
                       1: slide out downward,
                       2: slide out upward)

    V:      vibrato (0 for none,
                     1 for vibrato: vivrato with entext smaller or equal to 1 semitone,
                     2 for wild vibrato: vibrato with entext larger than 1 semitone)

"""

from mir_eval.transcription import precision_recall_f1
import numpy as np
import os, sys

def long_pattern_evaluate(pattern,bend_answer_path,slide_answer_path):
    if type(bend_answer_path).__name__=='ndarray':
        bend_answer = bend_answer_path.copy()
    else:
        bend_answer = np.loadtxt(bend_answer_path)
    if type(slide_answer_path).__name__=='ndarray':
        slide_answer = slide_answer_path.copy()
    else:
        slide_answer = np.loadtxt(slide_answer_path)    
    TP_bend = np.array([]);TP_slide = np.array([])
    FN_bend = np.array([]);FN_slide = np.array([])
    candidate = pattern.copy()
    candidate_mask = np.ones(len(candidate))
    bend_answer_mask = np.ones(len(bend_answer))
    slide_answer_mask = np.ones(len(slide_answer))  
    for c in range(len(candidate)):
        for b in range(len(bend_answer)):
            if bend_answer[b,0]>candidate[c,0] and bend_answer[b,0]<candidate[c,1]:
                candidate_mask[c] = 0
                bend_answer_mask[b] = 0
        for s in range(len(slide_answer)):
            if slide_answer[s,0]>candidate[c,0] and slide_answer[s,0]<candidate[c,1]:
                candidate_mask[c] = 0
                slide_answer_mask[s] = 0

    num_invalid_candidate = np.sum(candidate_mask)
    num_valid_candidate = len(candidate_mask)-num_invalid_candidate
    invalid_candidate = np.delete(candidate,np.nonzero(candidate_mask==0)[0],axis = 0)

    TP_bend = bend_answer[np.nonzero(bend_answer_mask==0)[0]]
    FN_bend = bend_answer[np.nonzero(bend_answer_mask==1)[0]]
    TP_slide = slide_answer[np.nonzero(slide_answer_mask==0)[0]]
    FN_slide = slide_answer[np.nonzero(slide_answer_mask==1)[0]]
    
    return num_valid_candidate, num_invalid_candidate, invalid_candidate, TP_bend, TP_slide, FN_bend, FN_slide


def short_pattern_evaluate(pattern,bend_answer_path,slide_answer_path,pullhamm_answer_path):
    if type(bend_answer_path).__name__=='ndarray':
        bend_answer = bend_answer_path.copy()
    else:
        bend_answer = np.loadtxt(bend_answer_path)
    if type(slide_answer_path).__name__=='ndarray':
        slide_answer = slide_answer_path.copy()
    else:
        slide_answer = np.loadtxt(slide_answer_path)
    if type(pullhamm_answer_path).__name__=='ndarray':
        pullhamm_answer = pullhamm_answer_path.copy()
    else:
        pullhamm_answer = np.loadtxt(pullhamm_answer_path)
    candidate = pattern.copy()
    candidate_mask = np.ones(len(candidate))
    bend_answer_mask = np.ones(len(bend_answer))
    slide_answer_mask = np.ones(len(slide_answer))
    pullhamm_answer_mask = np.ones(len(pullhamm_answer))
    for c in range(len(candidate)):
        for b in range(len(bend_answer)):
            if bend_answer[b,0]>candidate[c,0] and bend_answer[b,0]<candidate[c,1]:
                candidate_mask[c] = 0
                bend_answer_mask[b] = 0
        for s in range(len(slide_answer)):
            if slide_answer[s,0]>candidate[c,0] and slide_answer[s,0]<candidate[c,1]:
                candidate_mask[c] = 0
                slide_answer_mask[s] = 0
        for p in range(len(pullhamm_answer)):
            if pullhamm_answer[p,0]>candidate[c,0] and pullhamm_answer[p,0]<candidate[c,1]:
                candidate_mask[c] = 0
                pullhamm_answer_mask[p] = 0

    numInvalidCandidate = np.sum(candidate_mask)
    numValidCandidate = len(candidate_mask)-numInvalidCandidate
    InvalidCandidate = np.delete(candidate,np.nonzero(candidate_mask==0)[0],axis = 0)

    TP_bend = bend_answer[np.nonzero(bend_answer_mask==0)[0]]
    FN_bend = bend_answer[np.nonzero(bend_answer_mask==1)[0]]
    TP_slide = slide_answer[np.nonzero(slide_answer_mask==0)[0]]
    FN_slide = slide_answer[np.nonzero(slide_answer_mask==1)[0]]
    TP_pullhamm = pullhamm_answer[np.nonzero(pullhamm_answer_mask==0)[0]]
    FN_pullhamm = pullhamm_answer[np.nonzero(pullhamm_answer_mask==1)[0]]

    return numValidCandidate, numInvalidCandidate, InvalidCandidate, TP_bend, TP_slide, TP_pullhamm, FN_bend, FN_slide, FN_pullhamm

def fit_mir_eval_transcription(annotation, note):
    """
    Transform 2-D numpy array of note event into mir_eval format.
    Example:

                            P   On   Du
    annotation = np.array([[50, 1.2, 0.5]])

    Parameters
    ----------
    annotation: np.ndarray, shape=(n_event, 3)
        Annoatation of note event.
    note: np.ndarray, shape=(n_event, 3)
        Note event predictions.

    Returns
    -------
    ref_intervals: np.ndarray, shape=(n_event, 2)
    ref_pitches:   np.ndarray, shape=(n_event,)
    est_intervals: np.ndarray, shape=(n_event, 2)
    est_pitches:   np.ndarray, shape=(n_event,)

    """
    ref_intervals = np.vstack((annotation[:,1], annotation[:,1]+annotation[:,2])).T
    ref_pitches = annotation[:,0]
    est_intervals = np.vstack((note[:,1], note[:,1]+note[:,2])).T
    est_pitches = note[:,0]
    return ref_intervals, ref_pitches, est_intervals, est_pitches

def calculate_candidate_cls_accuracy_f_measure(annotation_ts_pseudo, candidate_result_pseudo, tech_index_dic):
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    # make pseudo answer and predicted labels
    annotation_ts = annotation_ts_pseudo.copy()
    candidate_result = candidate_result_pseudo.copy()
    # create answer list filled with index of normal
    y_true = np.empty([candidate_result.shape[0]])
    y_true.fill(tech_index_dic['normal'])
    # create predicted list
    y_pred = candidate_result[:,2].copy()
    answer_tech_dic = {'bend':[3,4,5], 'pull':[6], 'hamm':[7], 'slide':[8,9,10], 'vibrato':[11]}
    # make target tech list, e.g., ['bend', 'pull', 'vibrato']
    target_tech_list = [t for t in tech_index_dic if t in answer_tech_dic.keys()]
    # make target tech index list, e.g., [3,4,5,6,11]
    target_tech_index_list = []
    for t in target_tech_list:
        target_tech_index_list+=answer_tech_dic[t]

    # count technically candidate (all candidates - normal)
    tech_candidate = 0
    for index_candi, candi_result in enumerate(candidate_result):
        for index_ann, ts_ann in enumerate(annotation_ts):
            # if candidate time segment covers the instant of employment
            if candi_result[0] < ts_ann[0] and candi_result[1] > ts_ann[0]:
                # if the answer covered by candidate is in the target tech list
                if ts_ann[-1] in target_tech_index_list:
                    # fill answer list
                    if ts_ann[-1] in [3,4,5]:
                        y_true[index_candi]=tech_index_dic['bend']
                    elif ts_ann[-1]==6:
                        y_true[index_candi]=tech_index_dic['pull']
                    elif ts_ann[-1]==7:
                        y_true[index_candi]=tech_index_dic['hamm']
                    elif ts_ann[-1] in [8,9,10]:
                        y_true[index_candi]=tech_index_dic['slide']
                    elif ts_ann[-1]==11:
                        y_true[index_candi]=tech_index_dic['vibrato']
                    tech_candidate+=1
                else:
                    y_true[index_candi]=tech_index_dic['normal']

                # get key by value
                # t = [k for k, v in tech_index_dic.iteritems() if v == candi_result[-1]][0]

    # the ratio of (# corrected classified candidate / # of all candidates)
    cls_accuracy = accuracy_score(y_true, y_pred)
    # make target names list in index order
    target_names = []
    for index in range(len(tech_index_dic)):
        target_names.append([k for k, v in tech_index_dic.iteritems() if v == index][0])
    # make classification report
    cls_report = classification_report(y_true, y_pred, target_names=target_names)
    # make confusion matrix
    confusion_table = confusion_matrix(y_true, y_pred)

    # calculate non tech candidate which are predicted as normal
    # non_tech_candi_predicted_as_normal = np.where(candidate_result[np.where(candidate_result[:,2]!=-1)[0], 2]==tech_index_dic['normal'])[0].size
    # the ratio of (# of answers covered by candidate / # of all answers)
    candidate_answer_ratio = tech_candidate/float(annotation_ts.shape[0])
    # the ratio of (# of expression style candidates / # of all candidates)
    tech_candidte_ratio = tech_candidate/float(candidate_result.shape[0])

    return cls_accuracy, cls_report, confusion_table, candidate_answer_ratio, tech_candidte_ratio, target_names


def calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list):
    # check the annotation dimension
    try:
        annotation_ts.shape[1]
    except IndexError:
        annotation_ts = annotation_ts.reshape(1, annotation_ts.shape[0])
    # check the prediction dimension
    try:
        prediction_ts.shape[1]
    except IndexError:
        prediction_ts = prediction_ts.reshape(1, prediction_ts.shape[0])
    # check tech_index_list data type
    if type(tech_index_list)!=list:
        tech_index_list = [tech_index_list]
        
    (TP, FP, FN) = 0,0,0
    for tech_index in tech_index_list:
        target_annotation_ts = annotation_ts[np.where(annotation_ts[:,-1]==tech_index)[0],:]
        target_prediction_ts = prediction_ts[np.where(prediction_ts[:,-1]==tech_index)[0],:]
        (tp, fp, fn)=0,0,0
        for index_ann, ts_ann in enumerate(target_annotation_ts):
            for index_pre, ts_pre in enumerate(target_prediction_ts):
                if target_prediction_ts[index_pre,0]<target_annotation_ts[index_ann,0] and \
                   target_prediction_ts[index_pre,1]>target_annotation_ts[index_ann,0]:
                    target_annotation_ts[index_ann,-1]=-1
                    target_prediction_ts[index_pre,-1]=-1
        tp=np.where(target_prediction_ts[:,-1]==-1)[0].size
        fp=np.where(target_prediction_ts[:,-1]!=-1)[0].size
        fn=np.where(target_annotation_ts[:,-1]!=-1)[0].size
        TP=TP+tp
        FP=FP+fp
        FN=FN+fn

    # calculate precision, recall, f-measure
    if TP !=0 or FP!=0:
        P = TP/float(TP+FP) 
    elif TP ==0 and FP==0:
        P=0
    if TP !=0 or FN!=0:
        R = TP/float(TP+FN)
    elif TP ==0 and FN==0:
        R = 0
    if P !=0 or R!=0:
        F = 2*P*R/float(P+R)
    else:
        F=0

    return P, R, F, TP, FP, FN

def calculate_esn_f_measure(annotation_esn, prediction_esn, tech, onset_tolerance=0.05, offset_ratio=0.2, correct_pitch=None):
    annotation_esn_mask = annotation_esn.copy()
    prediction_esn_mask = prediction_esn.copy()
    if tech == 'Pre-bend':
        tech_index = 3
    elif tech == 'Bend':
        tech_index = 4
    elif tech == 'Release':
        tech_index = 5
    elif tech == 'Pull-off':
        tech_index = 6
    elif tech == 'Hammer-on':
        tech_index = 7
    elif tech == 'Slide':
        tech_index = 8
    elif tech == 'Slide in':
        tech_index = 9
    elif tech == 'Slide out':
        tech_index = 10
    elif tech == 'Vibrato':
        tech_index = 11

    if tech=='Pre-bend' or tech=='Bend' or tech=='Release' or tech=='Slide in' or tech=='Slide out' or tech=='Vibrato':
        # loop in annotated expression style note
        for index_ann, note_ann in enumerate(annotation_esn_mask):
            # if technique is identified in annotated esn
            if note_ann[tech_index]!=0:
                # loop in predicted expression style note
                for index_pre, note_pre in enumerate(prediction_esn_mask):
                    # if technique is identified in predicted esn
                    if note_pre[tech_index]!=0:
                        # check if two esn are matched
                        if note_ann[1]-onset_tolerance < note_pre[1] and note_ann[1]+onset_tolerance > note_pre[1]:
                            if correct_pitch is True and offset_ratio!=None:
                                if note_ann[0] == note_pre[0] and \
                                    note_ann[1]+note_ann[2]-note_ann[2]*offset_ratio < note_pre[1]+note_pre[2] and \
                                    note_ann[1]+note_ann[2]+note_ann[2]*offset_ratio > note_pre[1]+note_pre[2]:
                                    note_ann[tech_index] = -1
                                    note_pre[tech_index] = -1
                            elif correct_pitch is False and offset_ratio!=None:
                                if note_ann[1]+note_ann[2]-note_ann[2]*offset_ratio < note_pre[1]+note_pre[2] and \
                                    note_ann[1]+note_ann[2]+note_ann[2]*offset_ratio > note_pre[1]+note_pre[2]:
                                    note_ann[tech_index] = -1
                                    note_pre[tech_index] = -1
                            elif correct_pitch is True and offset_ratio==None:
                                if note_ann[0] == note_pre[0]:
                                    note_ann[tech_index] = -1
                                    note_pre[tech_index] = -1
                            elif correct_pitch is False and offset_ratio==None:
                                note_ann[tech_index] = -1
                                note_pre[tech_index] = -1


        TP = np.extract(annotation_esn_mask[:,tech_index]==-1, annotation_esn_mask[:,tech_index]).size
        FP = np.extract(prediction_esn_mask[:,tech_index]>0, prediction_esn_mask[:,tech_index]).size
        FN = np.extract(annotation_esn_mask[:,tech_index]>0, annotation_esn_mask[:,tech_index]).size

    elif tech=='Pull-off' or tech=='Hammer-on' or tech=='Slide':
        (TP, FP, FN)=0,0,0
        # loop in annotated expression style note
        for index_ann, note_ann in enumerate(annotation_esn_mask[:-1]):
            # if technique is identified in annotated esn
            if note_ann[tech_index]!=0 and annotation_esn_mask[index_ann+1,tech_index]!=0:
                # loop in predicted expression style note
                for index_pre, note_pre in enumerate(prediction_esn_mask[:-1]):
                    # if technique is identified in predicted esn
                    if note_pre[tech_index]!=0 and prediction_esn_mask[index_pre+1, tech_index]:
                        # check if two esn are matched
                        if annotation_esn_mask[index_ann+1,1]-onset_tolerance < prediction_esn_mask[index_pre+1,1] and \
                           annotation_esn_mask[index_ann+1,1]+onset_tolerance > prediction_esn_mask[index_pre+1,1]:
                            if correct_pitch is True and offset_ratio!=None:
                                if annotation_esn_mask[index_ann+1,0] == prediction_esn_mask[index_pre+1,0] and \
                                    annotation_esn_mask[index_ann+1,1]+annotation_esn_mask[index_ann+1,2]-annotation_esn_mask[index_ann+1,2]*offset_ratio < prediction_esn_mask[index_pre+1,1]+prediction_esn_mask[index_pre+1,2] and \
                                    annotation_esn_mask[index_ann+1,1]+annotation_esn_mask[index_ann+1,2]+annotation_esn_mask[index_ann+1,2]*offset_ratio > prediction_esn_mask[index_pre+1,1]+prediction_esn_mask[index_pre+1,2]:
                                    TP+=1
                                    annotation_esn_mask[index_ann,tech_index]=-1
                                    prediction_esn_mask[index_pre,tech_index]=-1    
                            elif correct_pitch is False and offset_ratio!=None:
                                if annotation_esn_mask[index_ann+1,1]+annotation_esn_mask[index_ann+1,2]-annotation_esn_mask[index_ann+1,2]*offset_ratio < prediction_esn_mask[index_pre+1,1]+prediction_esn_mask[index_pre+1,2] and \
                                    annotation_esn_mask[index_ann+1,1]+annotation_esn_mask[index_ann+1,2]+annotation_esn_mask[index_ann+1,2]*offset_ratio > prediction_esn_mask[index_pre+1,1]+prediction_esn_mask[index_pre+1,2]:
                                    TP+=1
                                    annotation_esn_mask[index_ann,tech_index]=-1
                                    prediction_esn_mask[index_pre,tech_index]=-1    
                            elif correct_pitch is True and offset_ratio==None:
                                if annotation_esn_mask[index_ann+1,0] == prediction_esn_mask[index_pre+1,0]:
                                    TP+=1
                                    annotation_esn_mask[index_ann,tech_index]=-1
                                    prediction_esn_mask[index_pre,tech_index]=-1
                            elif correct_pitch is False and offset_ratio==None:
                                TP+=1
                                annotation_esn_mask[index_ann,tech_index]=-1
                                prediction_esn_mask[index_pre,tech_index]=-1

        for index in range(prediction_esn_mask.shape[0]-1):
            if (prediction_esn_mask[index,tech_index]>0 and prediction_esn_mask[index+1,tech_index]>0) or \
               (prediction_esn_mask[index,tech_index]>0 and prediction_esn_mask[index+1,tech_index]<0):
                FP+=1

        for index in range(annotation_esn_mask.shape[0]-1):
            if (annotation_esn_mask[index,tech_index]>0 and annotation_esn_mask[index+1,tech_index]>0) or \
               (annotation_esn_mask[index,tech_index]>0 and annotation_esn_mask[index+1,tech_index]<0):
                FN+=1

    # calculate precision, recall, f-measure
    if TP !=0 or FP!=0:
        P = TP/float(TP+FP) 
    elif TP ==0 and FP==0:
        P=0
    if TP !=0 or FN!=0:
        R = TP/float(TP+FN)
    elif TP ==0 and FN==0:
        R = 0
    if P !=0 or R!=0:
        F = 2*P*R/float(P+R)
    else:
        F=0

    return P, R, F, TP, FP, FN

def evaluation_candidate_cls(annotation_ts, candidate_result, output_dir, filename, tech_index_dic, string=None, mode='a', plot=False):

    # evaluation
    (cls_accuracy, cls_report, confusion_table, 
     candidate_answer_ratio, tech_candidte_ratio, 
     target_names) = calculate_candidate_cls_accuracy_f_measure(annotation_ts, candidate_result, tech_index_dic=tech_index_dic)

    if plot:
        import matplotlib.pyplot as plt
        def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, tech_index_dic=tech_index_dic):
            tech_list=np.asarray(sorted(tech_index_dic.keys()))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(tech_list))
            plt.xticks(tick_marks, tech_list, rotation=45)
            plt.yticks(tick_marks, tech_list)
            # plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
        # Compute confusion matrix        
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(confusion_table)
        plt.savefig(output_dir+os.sep+filename+'.cm.png')
        plt.close('all')
        confusion_table_normalized = confusion_table.astype('float') / confusion_table.sum(axis=1)[:, np.newaxis]
        plt.figure()
        plot_confusion_matrix(confusion_table_normalized)
        plt.savefig(output_dir+os.sep+filename+'.norm.cm.png')
        plt.close('all')

    # write result to file
    save_stdout = sys.stdout
    fh = open(output_dir+os.sep+filename+'.cls.eval',mode)
    sys.stdout = fh
    if string:
        print string

    print '============================================================================'
    print 'Evaluation on song '+filename
    print '============================================================================'

    print '                      Candidate classification report                       '
    print '----------------------------------------------------------------------------'
    print 'Accuracy'
    print '--------'
    print '%8.4f'%cls_accuracy
    print ' '
    print 'The ratio of (# of answers covered by candidate / # of all answers)'
    print '-------------------------------------------------------------------'
    print '%8.4f'%candidate_answer_ratio
    print ' '
    print 'The ratio of (# of expression style candidates / # of all candidates)'
    print '---------------------------------------------------------------------'
    print '%8.4f'%tech_candidte_ratio
    print ' '
    print 'Classification report'
    print '---------------------'
    print cls_report
    print ' '
    print 'Confusion matrix'
    print '----------------'
    print '%8s'%' ',
    for t in target_names:
        if t!=target_names[-1]:
            print '%8s'%t,
        else:
            print '%8s'%t
    for index, row in enumerate(confusion_table):
        print '%8s'%target_names[index],
        for e in row:
            print '%8s'%e,
        print '\n'
    # return to normal:
    sys.stdout = save_stdout
    fh.close()


def evaluation_note(annotation, note, output_dir, filename, onset_tolerance=0.05, offset_ratio=0.2, string=None, mode='a'):    
    # convert format to fit mir_eval
    ref_intervals, ref_pitches, est_intervals, est_pitches = fit_mir_eval_transcription(annotation, note)
    save_stdout = sys.stdout
    fh = open(output_dir+os.sep+filename+'.note.eval',mode)
    sys.stdout = fh
    if string:
        print string
    print '======================================================================'
    print 'Note event evaluation for song '+filename
    print '======================================================================'

    print '                                 Note                                 '
    print '----------------------------------------------------------------------'
    print '                             Precision          Recall       F-measure'
    onset_tolerance=[0.05, 0.75, 0.1]
    offset_ratio=[0.20, 0.35, None]
    for on in onset_tolerance:
        for off in offset_ratio:
            note_p, note_r, note_f = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=on, offset_ratio=off)
            print ('%14s%16.4f%16.4f%16.4f' % ('CorPOn(%4ss)Off(%4s)', note_p, note_r, note_f) % (on, off))
    print '\n'

    sys.stdout = save_stdout
    fh.close()

def evaluation_esn(annotation_esn, prediction_esn, output_dir, filename, onset_tolerance=0.05, offset_ratio=0.2, string=None, mode='a'):

    # convert format to fit mir_eval
    ref_intervals, ref_pitches, est_intervals, est_pitches = fit_mir_eval_transcription(annotation_esn[:,0:3], prediction_esn[:,0:3])
    # write result to file
    # sys.stdout = open(output_dir+os.sep+filename+'.esn.eval', 'a')
    save_stdout = sys.stdout
    fh = open(output_dir+os.sep+filename+'.esn.eval',mode)
    sys.stdout = fh
    if string:
        print string

    print '======================================================================'
    print 'Evaluation on song '+filename
    print '======================================================================'

    print '                                 Note                                 '
    print '----------------------------------------------------------------------'
    print '                             Precision          Recall       F-measure'
    onset_tolerance=[0.05, 0.75, 0.1]
    offset_ratio=[0.20, 0.35, None]
    for on in onset_tolerance:
        for off in offset_ratio:
            note_p, note_r, note_f = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=on, offset_ratio=off)
            print ('%14s%16.4f%16.4f%16.4f' % ('CorPOn(%4ss)Off(%4s)', note_p, note_r, note_f) % (on, off))
    print '\n'

    onset_tolerance=[0.05, 0.1]
    offset_ratio=[0.20, 0.35, None]
    correct_pitch = [True, False]
    print '                       Expression style (note)                      '
    print '--------------------------------------------------------------------'
    for on in onset_tolerance:
        for off in offset_ratio:
            for cp in correct_pitch:
                print ('                Correct P(%s)On(%s)Off(%s)      ' % (cp, on, off))
                print '--------------------------------------------------------------------'
                print '               Precision                Recall             F-measure'
                P, R, F, TP, FP, FN = calculate_esn_f_measure(annotation_esn, prediction_esn, tech='Pre-bend', onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Pre-bend', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_esn_f_measure(annotation_esn, prediction_esn, tech='Bend', onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Bend', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_esn_f_measure(annotation_esn, prediction_esn, tech='Release', onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Release', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_esn_f_measure(annotation_esn, prediction_esn, tech='Pull-off', onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Pull-off', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_esn_f_measure(annotation_esn, prediction_esn, tech='Hammer-on', onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Hammer-on', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_esn_f_measure(annotation_esn, prediction_esn, tech='Slide', onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Slide', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_esn_f_measure(annotation_esn, prediction_esn, tech='Slide in', onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Slide in', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_esn_f_measure(annotation_esn, prediction_esn, tech='Slide out', onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Slide out', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_esn_f_measure(annotation_esn, prediction_esn, tech='Vibrato', onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Vibrato', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                print '                                                            '
    # return to normal:
    sys.stdout = save_stdout
    fh.close()


def evaluation_ts(annotation_ts, prediction_ts, output_dir, filename, string=None, mode='a'):

    save_stdout = sys.stdout
    fh = open(output_dir+os.sep+filename+'.ts.eval',mode)
    sys.stdout = fh
    if string:
        print string
    print '============================================================================'
    print 'Evaluation on song '+filename
    print '============================================================================'
    # 
    print '                    Expression style (time segment)                 '
    print '--------------------------------------------------------------------'
    print '               Precision                Recall             F-measure'
    
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=[3,4,5])
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Bend', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=6)
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Pull-off', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=7)
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Hammer-on', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=[8,9,10])
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Slide', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=11)
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Vibrato', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    print '                                                                            '



    print '--------------------------------------------------------------------'
    print '                     Techniques sub-divided                         '
    print '--------------------------------------------------------------------'
    print '               Precision                Recall             F-measure'
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=3)
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Pre-bend', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=4)
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Bend', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=5)
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Release', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=6)
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Pull-off', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=7)
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Hammer-on', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=8)
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Slide', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=9)
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Slide in', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=10)
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Slide out', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    P, R, F, TP, FP, FN = calculate_ts_f_measure(annotation_ts, prediction_ts, tech_index_list=11)
    print ('%12s%12.4f%10s%12.4f%10s%12.4s' % ('Vibrato', P ,' ('+str(TP)+'/'+str(TP+FP)+')', R, ' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
    print '                                                            '


    # return to normal:
    sys.stdout = save_stdout
    fh.close()





