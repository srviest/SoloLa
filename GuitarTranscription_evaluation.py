#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/04/24
--------------------------------------------------------------------------------
Script for evaluating performaces of guitar transcription
--------------------------------------------------------------------------------
"""

from mir_eval.transcription import precision_recall_f1
import numpy as np
import os, sys

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

def calculate_expr_f_measure(tech, annotation_esn, prediction_esn, onset_tolerance=0.05, offset_ratio=0.2, correct_pitch=None):
    annotation_esn_mask = annotation_esn.copy
    prediction_esn_mask = prediction_esn.copy

    if tech == 'Pre-bend':
        tech_index = 3
    elif tech == 'Bend':
        tech_index = 4
    elif tech == 'Release':
        tech_index = 5
    elif tech == 'Pull-off':
        tech_index = 6
    elif tech == 'Hmmaer-on':
        tech_index = 7
    elif tech == 'Slide':
        tech_index = 8
    elif tech == 'Slide in':
        tech_index = 9
    elif tech == 'Slide out':
        tech_index = 10
    elif tech == 'Vibrato':
        tech_index = 11

    # tech_num = np.count_nonzero(annotation_esn_mask[:,tech_index])

    # loop in annotated expression style note
    for index_ann, note_ann in enumerate(annotation_esn_mask):
        # if technique is identified in annotated esn
        if note_ann[tech_index]!=0:
            # loop in predicted expression style note
            for index_pre, note_pre in enumerate(prediction_esn_mask):
                # if technique is identified in predicted esn
                if note_pre[tech_index]!=0:
                    # check if two esn are matched
                    if note_ann[1]-onset_tolerance < note_pre[1] and note_ann[1]+onset_tolerance > note_pre[1] and \
                       note_ann[1]+note_ann[2]-note_ann[2]*offset_ratio < note_pre[1]+note_pre[2] and \
                       note_ann[1]+note_ann[2]+note_ann[2]*offset_ratio > note_pre[1]+note_pre[2]:
                        if correct_pitch is True:
                            if note_ann[0] == note_pre[0]:
                                note_ann[tech_index] = -1
                                note_pre[tech_index] = -1
                        elif:
                            note_ann[tech_index] = -1
                            note_pre[tech_index] = -1


    TP = np.extract(annotation_esn_mask[:,tech_index]==-1, annotation_esn_mask[:,tech_index]).size
    FP = np.extract(prediction_esn_mask[:,tech_index]>0, prediction_esn_mask[:,tech_index]).size
    FN = np.extract(annotation_esn_mask[:,tech_index]>0, annotation_esn_mask[:,tech_index]).size

    P = TP/float(TP+FP)
    R = TP/float(TP+FN)
    F = 2*P*R/float(P+R)

    return P, R, F, TP, FP, FN



def evaluation_note(annotation, note, output_dir, filename, onset_tolerance=0.05, offset_ratio=0.2, string=None):    
    # convert format to fit mir_eval
    ref_intervals, ref_pitches, est_intervals, est_pitches = fit_mir_eval_transcription(annotation, note)
    # ref_intervals, ref_pitches, pruned_est_intervals, pruned_est_pitches = fit_mir_eval_transcription(annotation, pruned_note)

    # evaluation
    p, r, f = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=offset_ratio)
    # pruned_p, pruned_r, pruned_f = precision_recall_f1(ref_intervals, ref_pitches, pruned_est_intervals, pruned_est_pitches, offset_ratio=0.2)

    # ignore offset
    p_ig_off, r_ig_off, f_ig_off = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=None)
    # pruned_p_ig_off, pruned_r_ig_off, pruned_f_ig_off = precision_recall_f1(ref_intervals, ref_pitches, pruned_est_intervals, pruned_est_pitches, offset_ratio=None)

    sys.stdout = open(output_dir+os.sep+filename+'.note.eval', 'w')
    if string:
        print string
    print '============================================================'
    print 'Note event evaluation for song '+filename
    print '============================================================'

    print 'Correct Pitch, Onset (%ss), Offset (%d%% of note length)' % (onset_tolerance, offset_ratio)
    print '------------------------------------------------------------'
    print '                   Precision          Recall       F-measure'
    print ('%12s%16.4f%16.4f%16.4f' % ('Raw note', p, r, f))
    # print ('%12s%16.4f%16.4f%16.4f' % ('Pruned note', pruned_p, pruned_r, pruned_f))
    print '============================================================'
    print 'Correct Pitch, Onset (%ss)' % (onset_tolerance)
    print '------------------------------------------------------------'
    print '                   Precision          Recall       F-measure'
    print ('%12s%16.4f%16.4f%16.4f' % ('Raw note', p_ig_off, r_ig_off, f_ig_off))
    # print ('%12s%16.4f%16.4f%16.4f' % ('Pruned note', pruned_p_ig_off, pruned_r_ig_off, pruned_f_ig_off))


def evaluation_expr(annotation_esn, prediction_esn, output_dir, filename, onset_tolerance=0.05, offset_ratio=0.2, string=None):

    # convert format to fit mir_eval
    ref_intervals, ref_pitches, est_intervals, est_pitches = fit_mir_eval_transcription(annotation_esn[:,0:3], prediction_esn[:,0:3])
    # calculate expression style f-measure
    P, R, F, TP, FP, FN = calculate_expr_f_measure(tech, annotation_esn, prediction_esn, onset_tolerance=onset_tolerance, offset_ratio=offset_ratio, correct_pitch=None)
    # write result to file
    sys.stdout = open(output_dir+os.sep+filename+'.esn.eval', 'a')
    if string:
        print string

    print '============================================================'
    print 'Evaluation on song '+filename
    print '============================================================'

    print '                            Note                            '
    print '------------------------------------------------------------'
    
    print '                   Precision          Recall       F-measure'
    note_p, note_r, note_f = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.05, offset_ratio=0.2)
    print ('%12s%16.4f%16.4f%16.4f' % ('CorPOn(%ss)Off(%s)', note_p, note_r, note_f) % (onset_tolerance, offset_ratio))
    note_p, note_r, note_f = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.075, offset_ratio=0.2)
    print ('%12s%16.4f%16.4f%16.4f' % ('CorPOn(%ss)Off(%s)', note_p, note_r, note_f) % (onset_tolerance, offset_ratio))
    note_p, note_r, note_f = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.1, offset_ratio=0.2)
    print ('%12s%16.4f%16.4f%16.4f' % ('CorPOn(%ss)Off(%s)', note_p, note_r, note_f) % (onset_tolerance, offset_ratio))
    note_p, note_r, note_f = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.05, offset_ratio=0.35)
    print ('%12s%16.4f%16.4f%16.4f' % ('CorPOn(%ss)Off(%s)', note_p, note_r, note_f) % (onset_tolerance, offset_ratio))
    note_p, note_r, note_f = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.075, offset_ratio=0.35)
    print ('%12s%16.4f%16.4f%16.4f' % ('CorPOn(%ss)Off(%s)', note_p, note_r, note_f) % (onset_tolerance, offset_ratio))
    note_p, note_r, note_f = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.1, offset_ratio=0.35)
    print ('%12s%16.4f%16.4f%16.4f' % ('CorPOn(%ss)Off(%s)', note_p, note_r, note_f) % (onset_tolerance, offset_ratio))
    note_p, note_r, note_f = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.05, offset_ratio=None)
    print ('%12s%16.4f%16.4f%16.4f' % ('CorPOn(%ss)Off(%s)', note_p, note_r, note_f) % (onset_tolerance, offset_ratio))
    note_p, note_r, note_f = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.075, offset_ratio=None)
    print ('%12s%16.4f%16.4f%16.4f' % ('CorPOn(%ss)Off(%s)', note_p, note_r, note_f) % (onset_tolerance, offset_ratio))
    note_p, note_r, note_f = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.1, offset_ratio=None)
    print ('%12s%16.4f%16.4f%16.4f' % ('CorPOn(%ss)Off(%s)', note_p, note_r, note_f) % (onset_tolerance, offset_ratio))
    print '\n'

    onset_tolerance=[0.05, 0.1]
    offset_ratio=[0.20, 35]
    correct_pitch = [True, False]
    print '                      Expression style                      '
    print '------------------------------------------------------------'
    for on in onset_tolerance:
        for off in offset_ratio:
            for cp in correct_pitch:
                print ('               (CorP(%s)On(%s)Off(%s))     ' % (cp, on, off))
                print '------------------------------------------------------------'
                print '                   Precision          Recall       F-measure'
                P, R, F, TP, FP, FN = calculate_expr_f_measure(tech='Pre-bend', annotation_esn, prediction_esn, onset_tolerance=on, onset_tolerance=off, correct_pitch=cp)
                print ('%12s%16.16s%16.16s%16.16s' % ('Pre-bend', str(P)+' ('+str(TP)+'/'+str(TP+FP)+')', str(R)+' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_expr_f_measure(tech='Bend', annotation_esn, prediction_esn, onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%16.16s%16.16s%16.16s' % ('Bend', str(P)+' ('+str(TP)+'/'+str(TP+FP)+')', str(R)+' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_expr_f_measure(tech='Release', annotation_esn, prediction_esn, onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%16.16s%16.16s%16.16s' % ('Release', str(P)+' ('+str(TP)+'/'+str(TP+FP)+')', str(R)+' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_expr_f_measure(tech='Pull-off', annotation_esn, prediction_esn, onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%16.16s%16.16s%16.16s' % ('Pull-off', str(P)+' ('+str(TP)+'/'+str(TP+FP)+')', str(R)+' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_expr_f_measure(tech='Hmmaer-on', annotation_esn, prediction_esn, onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%16.16s%16.16s%16.16s' % ('Hmmaer-on', str(P)+' ('+str(TP)+'/'+str(TP+FP)+')', str(R)+' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_expr_f_measure(tech='Slide in', annotation_esn, prediction_esn, onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%16.16s%16.16s%16.16s' % ('Slide in', str(P)+' ('+str(TP)+'/'+str(TP+FP)+')', str(R)+' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_expr_f_measure(tech='Slide out', annotation_esn, prediction_esn, onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%16.16s%16.16s%16.16s' % ('Slide out', str(P)+' ('+str(TP)+'/'+str(TP+FP)+')', str(R)+' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                P, R, F, TP, FP, FN = calculate_expr_f_measure(tech='Vibrato', annotation_esn, prediction_esn, onset_tolerance=on, offset_ratio=off, correct_pitch=cp)
                print ('%12s%16.16s%16.16s%16.16s' % ('Vibrato', str(P)+' ('+str(TP)+'/'+str(TP+FP)+')', str(R)+' ('+str(TP)+'/'+str(TP+FN)+')', str(F)))
                print '                                                            '



    # print '               Expression style (time segment)              '
    # print '------------------------------------------------------------'
    # print '                   Precision          Recall       F-measure'
    # print ('%12s%16.16s%16.16s%16.10s' % ('Pre-bend', pb_p, pb_r, pb_f))
    # print ('%12s%16.16s%16.16s%16.10s' % ('Bend', b_p, b_r, b_f))
    # print ('%12s%16.16s%16.16s%16.10s' % ('Release', r_p, r_r, r_f))
    # print ('%12s%16.16s%16.16s%16.10s' % ('Pull-of', p_p, p_r, p_f))
    # print ('%12s%16.16s%16.16s%16.10s' % ('Hmmaer-on', h_p, h_r, h_f))
    # print ('%12s%16.16s%16.16s%16.10s' % ('Slide', s_p, s_r, s_f))
    # print ('%12s%16.16s%16.16s%16.10s' % ('Slide in', si_p, si_r, si_f))
    # print ('%12s%16.16s%16.16s%16.10s' % ('Slide out', so_p, so_r, so_f))
    # print ('%12s%16.16s%16.16s%16.10s' % ('Vibrato', v_p, v_r, v_f))
    









