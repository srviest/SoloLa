"""
Author: Yuan-Ping Chen, Ting-Wei Su
Date: 2016/04/24
--------------------------------------------------------------------------------
Functions for evaluating performaces of guitar transcription
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

from mir_eval.transcription import precision_recall_f1_overlap
from mir_eval.onset import f_measure
from technique import *
import numpy as np
import os, sys, csv

def fit_mir_eval_transcription(ans_list, pred_list):
    """
    Transform array of note events into mir_eval format.

    Parameters
    ----------
    ans_list: np.ndarray, shape=(n_event, 3)
        Answer of note events.
    pred_list: np.ndarray, shape=(n_event, 3)
        Prediction of note events.
    
    Returns
    -------
    ref_intervals: np.ndarray, shape=(n_event, 2)
    ref_pitches:   np.ndarray, shape=(n_event,)
    est_intervals: np.ndarray, shape=(n_event, 2)
    est_pitches:   np.ndarray, shape=(n_event,)
    """
    ref_intervals = np.array([[esn.onset, esn.offset] for esn in ans_list])
    ref_pitches = np.array([esn.pitch for esn in ans_list])
    est_intervals = np.array([[esn.onset, esn.offset] for esn in pred_list])
    est_pitches = np.array([esn.pitch for esn in pred_list])
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
    answer_tech_dic = {'bend':[4,5], 'pull':[6], 'hamm':[7], 'slide':[8,9,10], 'vibrato':[11]}
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
                    if ts_ann[-1] in [4,5]:
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


def calculate_ts_f_measure(annotation_ts, prediction_ts, tech):
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
        
    ans_ts_list = annotation_ts[np.where(annotation_ts[:,-1]==tech)[0],:]
    pred_ts_list = prediction_ts[np.where(prediction_ts[:,-1]==tech)[0],:]
    TP, FP, FN = 0,0,0
    a_i, p_i = 0, 0
    while a_i < len(ans_ts_list) and p_i < len(pred_ts_list):
        ans, pred = ans_ts_list[a_i], pred_ts_list[p_i]
        if ans[0] < pred[0]: a_i += 1; continue;
        if ans[0] > pred[1]: p_i += 1; continue;
        TP += 1
        a_i += 1
        p_i += 1
    FP = len(pred_ts_list) - TP
    FN = len(ans_ts_list) - TP

    # calculate precision, recall, f-measure
    P = TP/float(TP+FP) if (TP !=0 or FP!=0) else 0 
    R = TP/float(TP+FN) if (TP !=0 or FN!=0) else 0
    F = 2*P*R/float(P+R) if (P !=0 or R!=0) else 0

    return P, R, F, TP, FP, FN

def calculate_esn_f_measure(ans_list, pred_list, tech, onset_tolerance=0.05, offset_ratio=None, correct_pitch=True):
    def check_condition(a_i, p_i):
        ans, pred = ans_list[a_i], pred_list[p_i]
        # Check onset correctness
        if pred.onset < ans.onset - onset_tolerance: return False, a_i, p_i + 1
        if pred.onset > ans.onset + onset_tolerance: return False, a_i + 1, p_i
        # Check tech correctness
        tech_cond = ans.equal_tech(pred) if tech is None else (ans.tech(tech).value == pred.tech(tech).value > 0)
        if not tech_cond:
            (a_i, p_i) = (a_i, p_i + 1) if pred.onset < ans.onset else (a_i + 1, p_i)
            return False, a_i, p_i
        # Check pitch and offset correctness if needed
        correct_pitch_cond = (ans.pitch == pred.pitch) if correct_pitch == True else True
        offset_ratio_cond = (ans.offset - ans.duration*offset_ratio < pred.offset < ans.offset + ans.duration*offset_ratio) \
                            if offset_ratio is not None else True
        if correct_pitch_cond and offset_ratio_cond: return True, a_i+1, p_i+1
        else: 
            (a_i, p_i) = (a_i, p_i + 1) if pred.onset < ans.onset else (a_i + 1, p_i)
            return False, a_i, p_i

    def count_tech_in_list(esn_list, tch):
        idx, ct = 0, 0
        while idx < len(esn_list):
            esn = esn_list[idx]
            if esn.tech(tch).value > 0:
                if tech in [T_PULL, T_HAMMER, T_SLIDE]:
                    if idx + 1 < len(esn_list) and esn_list[idx+1].tech(tch).value > 0: 
                        ct += 1
                        if esn_list[idx+1].tech(tch).value == 2:
                            idx += 1
                else: ct += 1
            idx += 1
        return ct

    TP, FP, FN = 0, 0, 0
    a_i, p_i = 0, 0
    while a_i < len(ans_list) and p_i < len(pred_list):
        correct, a_i, p_i = check_condition(a_i, p_i)
        if correct and tech in [T_PULL, T_HAMMER, T_SLIDE] and \
           a_i + 1 < len(ans_list) and p_i + 1 < len(pred_list):
            correct, a_i, p_i = check_condition(a_i + 1, p_i + 1)
        if correct: 
            TP += 1
    tch_list = range(T_PREBEND, T_NORMAL) if tech is None else [tech]
    n_pred_techs, n_ans_techs = 0, 0
    for tch in tch_list:
        n_pred_techs += count_tech_in_list(pred_list, tch)
        n_ans_techs += count_tech_in_list(ans_list, tch)
    FP = n_pred_techs - TP
    FN = n_ans_techs - TP
    print tech, TP, n_pred_techs, n_ans_techs

    # calculate precision, recall, f-measure
    P = TP/float(TP+FP) if (TP !=0 or FP!=0) else 0
    R = TP/float(TP+FN) if (TP !=0 or FN!=0) else 0
    F = 2*P*R/float(P+R) if (P !=0 or R!=0) else 0

    return P, R, F, TP, FP, FN


def evaluation_candidate_cls(annotation_ts_orig, candidate_result_orig, output_dir, filename, tech_index_dic, string=None, poly_mask=None, mode='a', plot=True):

    if poly_mask:
        poly_mask = np.loadtxt(poly_mask)
        annotation_ts = remove_poly_ts(annotation_ts_orig, poly_mask)
        candidate_result = remove_poly_ts(candidate_result_orig, poly_mask)
    else:
        annotation_ts = annotation_ts_orig
        candidate_result = candidate_result_orig

    # evaluation
    (cls_accuracy, cls_report, confusion_table, 
     candidate_answer_ratio, tech_candidte_ratio, 
     target_names) = calculate_candidate_cls_accuracy_f_measure(annotation_ts, candidate_result, tech_index_dic=tech_index_dic)

    if plot:
        import matplotlib.pyplot as plt
        def plot_confusion_matrix(cm, output_path, title='Confusion matrix', cmap=plt.cm.Blues, tech_index_dic=tech_index_dic):
            np.set_printoptions(precision=2)
            plt.figure()
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
            plt.savefig(output_path)
        # Compute confusion matrix
        plot_confusion_matrix(confusion_table, title='Confusion matrix', output_path=output_dir+os.sep+filename+'.cm.png')
        confusion_table_normalized = confusion_table.astype('float') / confusion_table.sum(axis=1)[:, np.newaxis]        
        plot_confusion_matrix(confusion_table_normalized, title='Normalized confusion matrix', output_path=output_dir+os.sep+filename+'.norm.cm.png')


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

def remove_poly_notes(notes, poly_mask):
    notes_poly_removed = notes.copy()
    note_to_be_deleted = []
    for index_n, note in enumerate(notes):
        onset = note[1]
        offset = note[1]+note[2]
        for index_p, poly_note in enumerate(poly_mask):
            onset_in = poly_note[0] < onset < poly_note[1]
            offset_in = poly_note[0] < offset < poly_note[1]
            if onset_in or offset_in:
                note_to_be_deleted.append(index_n)
    notes_poly_removed = np.delete(notes_poly_removed, note_to_be_deleted, axis=0)
    return notes_poly_removed


def evaluation_note(ans_list, pred_list, output_dir, filename, 
                    onset_tolerance=0.05, offset_ratio=0.2, 
                    string=None, mode='a', verbose=False, 
                    separator=' ', poly_mask=None, extension=''):
    if poly_mask:
        poly_mask = np.loadtxt(poly_mask)
        ans_list = remove_poly_notes(ans_list, poly_mask)
        pred_list = remove_poly_notes(pred_list, poly_mask)
    ref_intervals, ref_pitches, est_intervals, est_pitches = fit_mir_eval_transcription(ans_list, pred_list)
    result = []

    # result with onset, pitch, offset on
    onset_tolerance=0.05
    offset_ratio=0.20
    note_p, note_r, note_f, avg_or = precision_recall_f1_overlap(ref_intervals, ref_pitches, 
                                                                 est_intervals, est_pitches, 
                                                                 onset_tolerance=onset_tolerance, 
                                                                 offset_ratio=offset_ratio)
    result += [round(note_p, 5)]+[round(note_r, 5)]+[round(note_f, 5)]+['']

    # result with onset, pitch on
    onset_tolerance=0.05
    offset_ratio=None
    note_p, note_r, note_f, avg_or = precision_recall_f1_overlap(ref_intervals, ref_pitches, 
                                                                 est_intervals, est_pitches, 
                                                                 onset_tolerance=onset_tolerance, 
                                                                 offset_ratio=offset_ratio)
    result += [round(note_p, 5)]+[round(note_r, 5)]+[round(note_f, 5)]+['']

    # result with onset on
    onset_f, onset_p, onset_r = f_measure(ref_intervals[:,0], est_intervals[:,0], window=onset_tolerance)
    result += [round(onset_p, 5)]+[round(onset_r, 5)]+[round(onset_f, 5)]+['']

    if string: result += [string]
    fh = open(output_dir+os.sep+filename+'.note.eval'+extension, mode)
    w = csv.writer(fh, delimiter = ',')
    if mode == 'w': 
        w.writerow(['C_On_P_Off','','','','C_On_P','','','','C_On','','','','Stage'])
    w.writerow(result)
    fh.close()
    print result

def remove_poly_esn(esn_list, poly_mask):
    esn_poly_removed = esn_list.copy()
    esn_to_be_deleted = []
    for index_n, esn in enumerate(esn_list):
        onset = esn.onset
        offset = esn.offset
        for index_p, poly_note in enumerate(poly_mask):
            onset_in = poly_note[0] < onset < poly_note[1]
            offset_in = poly_note[0] < offset < poly_note[1]
            if onset_in or offset_in:
                esn_to_be_deleted.append(index_n)
    esn_poly_removed = np.delete(esn_poly_removed, esn_to_be_deleted, axis=0)
    return esn_poly_removed


def evaluation_esn( ans_list, pred_list, output_dir, filename, 
                    onset_tolerance=0.05, offset_ratio=0.2, 
                    poly_mask=None, string=None, mode='a', 
                    extension=''):
    def append_result(data, type_str, P, R, F, TP, FP, FN):
        data.append([type_str, round(P, 4) ,' ('+str(TP)+'/'+str(TP+FP)+')', round(R, 4), ' ('+str(TP)+'/'+str(TP+FN)+')', round(F, 4)])

    def combine_evals(res_dict, tech_list):
        result_lists = [res_dict[t] for t in tech_list]
        TP, FP, FN = 0, 0, 0
        TP = sum([res[3] for res in result_lists])
        FP = sum([res[4] for res in result_lists])
        FN = sum([res[5] for res in result_lists])
        P = TP/float(TP+FP) if (TP != 0 or FP != 0) else 0
        R = TP/float(TP+FN) if (TP != 0 or FN != 0) else 0
        F = 2*P*R/float(P+R) if (P != 0 or R != 0) else 0
        return P, R, F, TP, FP, FN

    if poly_mask:
        poly_mask = np.loadtxt(poly_mask)
        ans_list = remove_poly_esn(ans_list, poly_mask)
        pred_list = remove_poly_esn(pred_list, poly_mask)

    data = []
    if string: data.append([string])

    ref_intervals, ref_pitches, est_intervals, est_pitches = fit_mir_eval_transcription(ans_list, pred_list)
    onset_tolerance=0.05
    offset_ratio=None
    note_p, note_r, note_f, avg_or = precision_recall_f1_overlap(ref_intervals, ref_pitches, 
                                                                 est_intervals, est_pitches, 
                                                                 onset_tolerance=onset_tolerance, 
                                                                 offset_ratio=offset_ratio)
    data.append(['C_On_P_Off', round(note_p, 4) ,'', round(note_r, 4), '', round(note_f, 4)])

    eval_dict = {T_NORMAL:[], T_PREBEND:[], T_BEND:[], 
                 T_RELEASE:[], T_PULL:[], T_HAMMER:[], 
                 T_SLIDE:[], T_SLIDE_IN:[], T_SLIDE_OUT:[], T_VIBRATO:[]}
    for tech in eval_dict:
        eval_dict[tech] = list(calculate_esn_f_measure(ans_list, pred_list, tech=tech, onset_tolerance=0.05))
        append_result(data, T_STR_DICT[tech], *(eval_dict[tech]))

    P, R, F, TP, FP, FN = calculate_esn_f_measure(ans_list, pred_list, tech=None, onset_tolerance=0.05)
    append_result(data, 'All', P, R, F, TP, FP, FN)

    data.append(['---------------------------------------'])

    all_bend_res = combine_evals(eval_dict, [T_PREBEND, T_BEND, T_RELEASE])
    append_result(data, T_STR_DICT[T_BEND], *all_bend_res)
    append_result(data, T_STR_DICT[T_PULL], *(eval_dict[T_PULL]))
    append_result(data, T_STR_DICT[T_HAMMER], *(eval_dict[T_HAMMER]))
    all_slide_res = combine_evals(eval_dict, [T_SLIDE, T_SLIDE_IN, T_SLIDE_OUT])
    append_result(data, T_STR_DICT[T_SLIDE], *all_slide_res)
    append_result(data, T_STR_DICT[T_VIBRATO], *(eval_dict[T_VIBRATO]))
    data.append([''])

    # write results
    fh = open(output_dir+os.sep+filename+'.esn.eval'+extension,mode)
    w = csv.writer(fh, delimiter = ',')
    if mode=='w': w.writerow(['', 'Precision', '', 'Recall', '',  'F-measure'])
    for r in data: 
        w.writerow(r)
        print r
    fh.close()

def remove_poly_ts(ts, poly_mask):
    ts_poly_removed = ts.copy()
    # check the ts dimension
    try:
        ts_poly_removed.shape[1]
    except IndexError:
        ts_poly_removed = ts_poly_removed.reshape(1, ts_poly_removed.shape[0])
    
    ts_to_be_deleted = []
    # remove predicted techniques
    if ts_poly_removed.shape[1]==3:
        for index_ts, ts in enumerate(ts_poly_removed):
            start_time = ts[0]
            end_time = ts[1]
            for index_p, poly_note in enumerate(poly_mask):
                start_time_in = poly_note[0] < start_time < poly_note[1]
                end_time_in = poly_note[0] < end_time < poly_note[1]
                if start_time_in or end_time_in:
                    ts_to_be_deleted.append(index_ts)
        
    # remove annotated techniques
    elif ts_poly_removed.shape[1]==2:
        for index_ts, ts in enumerate(ts_poly_removed):
            start_time = ts[0]
            for index_p, poly_note in enumerate(poly_mask):
                start_time_in = poly_note[0] < start_time < poly_note[1]
                if start_time_in:
                    ts_to_be_deleted.append(index_ts)

    ts_poly_removed = np.delete(ts_poly_removed, ts_to_be_deleted, axis=0)

    return ts_poly_removed


def evaluation_ts(ans_ts_list, pred_ts_list, output_dir, 
                  filename, string=None, mode='a', 
                  poly_mask=None, extension=''):
    def append_result(data, type_str, P, R, F, TP, FP, FN):
        data.append([type_str, round(P, 4) ,' ('+str(TP)+'/'+str(TP+FP)+')', round(R, 4), ' ('+str(TP)+'/'+str(TP+FN)+')', round(F, 4)])

    def combine_evals(res_dict, tech_list):
        result_lists = [res_dict[t] for t in tech_list]
        TP, FP, FN = 0, 0, 0
        TP = sum([res[3] for res in result_lists])
        FP = sum([res[4] for res in result_lists])
        FN = sum([res[5] for res in result_lists])
        P = TP/float(TP+FP) if (TP !=0 or FP!=0) else 0
        R = TP/float(TP+FN) if (TP !=0 or FN!=0) else 0
        F = 2*P*R/float(P+R) if (P !=0 or R!=0) else 0

    if poly_mask:
        poly_mask = np.loadtxt(poly_mask)
        ans_ts_list = remove_poly_ts(ans_ts_list, poly_mask)
        pred_ts_list = remove_poly_ts(pred_ts_list, poly_mask)

    data = []
    if string: data.append([string])

    eval_dict = {T_NORMAL:[], T_PREBEND:[], T_BEND:[], 
                 T_RELEASE:[], T_PULL:[], T_HAMMER:[], 
                 T_SLIDE:[], T_SLIDE_IN:[], T_SLIDE_OUT:[], T_VIBRATO:[]}
    for tech in eval_dict:
        eval_dict[tech] = calculate_ts_f_measure(ans_ts_list, pred_ts_list, tech)
        append_result(data, T_STR_DICT[tech], *(eval_dict[tech]))

    data.append(['---------------------------------------'])

    all_bend_res = combine_evals(eval_dict, [T_PREBEND, T_BEND, T_RELEASE])
    append_result(data, T_STR_DICT[T_BEND], *all_bend_res)
    append_result(data, T_STR_DICT[T_PULL], *(eval_dict[T_PULL]))
    append_result(data, T_STR_DICT[T_HAMMER], *(eval_dict[T_HAMMER]))
    all_slide_res = combine_evals(eval_dict, [T_SLIDE, T_SLIDE_IN, T_SLIDE_OUT])
    append_result(data, T_STR_DICT[T_SLIDE], *all_slide_res)
    append_result(data, T_STR_DICT[T_VIBRATO], *(eval_dict[T_VIBRATO]))
    data.append([''])

    # write results
    fh = open(output_dir+os.sep+filename+'.ts.eval'+extension,mode)
    w = csv.writer(fh, delimiter = ',')
    if mode=='w': w.writerow(['', 'Precision', '', 'Recall', '',  'F-measure'])
    for r in data: w.writerow(r)
    fh.close()
    