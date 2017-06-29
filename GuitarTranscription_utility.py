#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/04/24
--------------------------------------------------------------------------------
Utilities of guitar transcription.
--------------------------------------------------------------------------------
"""
import numpy as np

def hertz2midi(melody_contour):
    """
    Convert pitch sequence from hertz to MIDI scale.

    :param melody_contour: array of pitch sequence.
    :returns             : melody contour in MIDI scale.

    """ 
    from numpy import inf
    melody_contour_MIDI = melody_contour.copy()
    melody_contour_MIDI = np.log(melody_contour_MIDI/float(440))
    melody_contour_MIDI =12*melody_contour_MIDI/np.log(2)+69
    melody_contour_MIDI[melody_contour_MIDI==-inf]=0

    return melody_contour_MIDI


def midi2hertz(melody_contour):
    """
    convert pitch sequence from hertz to midi scale.
    :param melody_contour: array of pitch sequence.

    """ 
    from numpy import inf
    # melody_contour_Hz = melody_contour.copy()
    melody_contour_Hz = (melody_contour-69)*np.log(2)/12
    melody_contour_Hz = np.exp(melody_contour_Hz)*440
    redundant=np.exp((0-69)*np.log(2)/12)*440
    melody_contour_Hz[melody_contour_Hz==redundant]=0

    return melody_contour_Hz

def note_pruning(note_pseudo, threshold=0.1):
    """
    prune the note whose length smaller than the threshold.

    :param threshold: the minimal duration of note 

    """
    note = note_pseudo.copy()
    pruned_notes = np.empty([0,note.shape[1]])
    for n in range(note.shape[0]):
        if note[n,2]>threshold:
            pruned_notes = np.append(pruned_notes,[note[n,:]],axis=0)
    return pruned_notes

