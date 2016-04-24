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


def note_evaluation(annotation, note, pruned_note, output_dir, filename, onset_tolerance=0.05, offset_ratio=0.2):    
    ref_intervals, ref_pitches, est_intervals, est_pitches = fit_mir_eval_transcription(annotation, note)
    ref_intervals, ref_pitches, pruned_est_intervals, pruned_est_pitches = fit_mir_eval_transcription(annotation, pruned_note)
    precision, recall, f_measure = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=0.2)
    pruned_precision, pruned_recall, pruned_f_measure = precision_recall_f1(ref_intervals, ref_pitches, pruned_est_intervals, pruned_est_pitches, offset_ratio=0.2)

    precision_ig_off, recall_ig_off, f_measure_ig_off = precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=None)
    pruned_precision_ig_off, pruned_recall_ig_off, pruned_f_measure_ig_off = precision_recall_f1(ref_intervals, ref_pitches, pruned_est_intervals, pruned_est_pitches, offset_ratio=None)

    sys.stdout = open(output_dir+os.sep+filename+'.note.eval', 'w')
    print '============================================================'
    print 'Note event evaluation for song '+filename
    print '============================================================'

    print 'Correct Pitch, Onset (%ss), Offset (%d%% of note length)' % (onset_tolerance, offset_ratio*100)
    print '------------------------------------------------------------'
    print '                   Precision          Recall       F-measure'
    print ('%12s%16.4f%16.4f%16.4f' % ('Raw note', precision, recall, f_measure))
    print ('%12s%16.4f%16.4f%16.4f' % ('Pruned note', pruned_precision, pruned_recall, pruned_f_measure))
    print '============================================================'
    print 'Correct Pitch, Onset (%ss)' % (onset_tolerance)
    print '------------------------------------------------------------'
    print '                   Precision          Recall       F-measure'
    print ('%12s%16.4f%16.4f%16.4f' % ('Raw note', precision_ig_off, recall_ig_off, f_measure_ig_off))
    print ('%12s%16.4f%16.4f%16.4f' % ('Pruned note', pruned_precision_ig_off, pruned_recall_ig_off, pruned_f_measure_ig_off))


