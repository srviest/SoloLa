#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/03/10
-------------------------------------------------------------------------------
Expression style recognition: automatically recognize the electric 
                              guitar expression style.
-------------------------------------------------------------------------------
Args:
    input_files:    Audio files to be processed. 
                    Only the wav files would be considered.



    output_dir:     Directory for storing the results.

Optional args:
    Please refer to --help.
-------------------------------------------------------------------------------
Returns:
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

import glob, os
import numpy as np
import subprocess as subp
import operator
import pickle
from essentia import *
from essentia.standard import *
from Candidate_selection import continuously_ascending_descending_pattern, candidate_selection
from Feature_extraction import feature_extractor
from Classification import data_preprocessing
from GuitarTranscription_parameters import *
from GuitarTranscription_utility import note_pruning
from GuitarTranscription_evaluation import evaluation_note, evaluation_expr, evaluation_candidate_classification
import fnmatch

class Common(object):
    @staticmethod
    def update(expression_style_note, note_with_expression_style, technique, sub_technique):
        """
        Update expression_style_note array.

        :param expression_style_note:       numpy array of expression_style_note.
        :param note_with_expression_style:  numpy array of note event with expression style.
        :param technique:                   string of technique.
        :param sub_technique:               float number of sub technique.
        :returns:                           numpy array of updated expression_style_note.
        """
        if technique=='pre-bend': t = 3
        elif technique=='bend': t = 4
        elif technique=='release': t = 5
        elif technique=='pull': t = 6
        elif technique=='hamm': t = 7
        elif technique=='slide': t = 8
        elif technique=='slide-in': t = 9
        elif technique=='slide-out': t = 10
        elif technique=='vibrato': t = 11

        note_to_be_deleted = np.empty([0])
        for r_n in range(len(note_with_expression_style)):
            for r_esn in range(len(expression_style_note)):
                # if the onsets of expression_style_note and note_with_expression_style are equal
                if note_with_expression_style[r_n,1]==expression_style_note[r_esn,1]:   
                    # if the duration of current expression_style_note is larger than or equal to the duration of note_with_expression_style 
                    if expression_style_note[r_esn,2]>=note_with_expression_style[r_n,2]:
                        expression_style_note[r_esn,2]=note_with_expression_style[r_n,2]
                        expression_style_note[r_esn,t]=sub_technique
                    else:
                        # loop from the next expression_style_note
                        for r_esn_r in range(r_esn+1,len(expression_style_note)):
                            expression_style_note_offset = expression_style_note[r_esn_r,1]+expression_style_note[r_esn_r,2]
                            note_with_expression_style_offset = note_with_expression_style[r_n,1]+note_with_expression_style[r_n,2]
                            # check if the offset of expression_style_note is larger than or equal to the offset of note_with_expression_style
                            if expression_style_note_offset>=note_with_expression_style_offset:
                                # the expression_style_note will not be deleted if the onset exceed the offset of note_with_expression_style, vice versa
                                if expression_style_note[r_esn_r,1]>note_with_expression_style_offset:
                                    expression_style_note[r_esn,2]=note_with_expression_style[r_n,2]
                                    expression_style_note[r_esn,t]=sub_technique
                                    break
                                else:
                                    expression_style_note[r_esn,2]=note_with_expression_style[r_n,2]
                                    expression_style_note[r_esn,t]=sub_technique
                                    note_to_be_deleted = np.append(note_to_be_deleted,[r_esn_r], axis=0)
                                    break
                            else:
                                note_to_be_deleted = np.append(note_to_be_deleted,[r_esn_r], axis=0)

        expression_style_note = np.delete(expression_style_note, note_to_be_deleted,axis=0)
        return expression_style_note

    def sec_2_note(self):
        self.long_slide = np.empty([0,3])
        for r_lss in range(len(self.long_slide_sec)):
            for r_mn in range(len(self.merged_note)):
                if self.long_slide_sec[r_lss,0]>self.merged_note[r_mn,1] and self.long_slide_sec[r_lss,0]<self.merged_note[r_mn,1]+self.merged_note[r_mn,2]:
                    long_slide_pitch = self.merged_note[r_mn,0]
                    long_slide_onset = self.merged_note[r_mn,1]
                    if r_mn+1<=len(self.merged_note):
                        for r_mn_r in range(r_mn+1,len(self.merged_note)):
                            if self.long_slide_sec[r_lss,1]>self.merged_note[r_mn_r,1] and self.long_slide_sec[r_lss,1]<self.merged_note[r_mn_r,1]+self.merged_note[r_mn_r,2]:
                                long_slide_offset = self.merged_note[r_mn_r,1]+self.merged_note[r_mn_r,2]
                            else:
                                long_slide_offset = self.long_slide_sec[r_lss,1]
                    else:
                        long_slide_offset = self.long_slide_sec[r_lss,1]
                    long_slide = [long_slide_pitch, long_slide_onset, long_slide_offset]
                    self.long_slide = np.append(self.long_slide, [long_slide], axis=0)  

class WildVibrato(Common):

    def __init__(self):
        """
        Creates a new Wav object instance of the given file.

        :param filename: name of the .wav file

        """
        # self.merged_note = merged_note.copy()
        self.technique = 'vibrato'
        self.raw_note = None

    def detect(self, raw_note):
        self.raw_note = raw_note
        merged_notes, self.super_wild_vibrato = WildVibrato.identify_serrated_pattern(self.raw_note,2)
        # vibrato with extent of 1 semitone
        merged_notes, self.wild_vibrato = WildVibrato.identify_serrated_pattern(merged_notes,1)

        expression_style_note = np.hstack((merged_notes,np.zeros((merged_notes.shape[0],9))))

        expression_style_note = Common.update(expression_style_note=expression_style_note, 
                                              note_with_expression_style=self.super_wild_vibrato, 
                                              technique=self.technique, 
                                              sub_technique=2)

        expression_style_note = Common.update(expression_style_note=expression_style_note, 
                                              note_with_expression_style=self.wild_vibrato, 
                                              technique=self.technique, 
                                              sub_technique=2)
        return expression_style_note

    @staticmethod
    def identify_serrated_pattern(note_pseudo,extent):
        """
        Merge notes of wild vibrato by merging series of notes in serrated patter 
        Usage:
        :param note:     array of notes [pitch(MIDI#) onset(sec) duration(sec)].
        :param extent:   the heigh in semitone of the serrated pattern.
        :returns:        merged notes.
                         wild vibrato notes.         

        """
        note = note_pseudo.copy()
        wild_vibrato = np.empty([0,3])
        merged_notes = np.empty([0,3])
        for n in range(note.shape[0]):
            # if the pitch of current note is not zero
            if note[n,0]!=0 and n+1<=note.shape[0]-1:
                # the absolute pitch difference of current note and next note is a semitone:
                # the gap of current and next note is smaller than 0.01 seconds
                if note[n+1,0]-note[n,0]==extent and note[n+1,1]-(note[n,1]+note[n,2])<0.01:
                    pitch = note[n,0]
                    pitch_next = note[n+1,0]
                    onset_note = n
                    offset_note = n+1
                    sign = np.sign(pitch_next-pitch)
                    if offset_note+1<=note.shape[0]-1:
                        while( abs(note[offset_note+1,0]-note[offset_note,0])==extent and \
                            np.sign(note[offset_note+1,0]-note[offset_note,0]) != sign and \
                            note[offset_note+1,1]-(note[offset_note,1]+note[offset_note,2])<0.01 and \
                            offset_note+1<note.shape[0]-1):
                            sign = np.sign(note[offset_note+1,0]-note[offset_note,0])
                            if offset_note+1<note.shape[0]-1:
                                offset_note = offset_note+1
                            else:
                                break
                    num_notes = offset_note-onset_note+1
                    if num_notes>=5:                
                        onset_time = note[onset_note,1]
                        duration = note[offset_note,1]+note[offset_note,2]-onset_time
                        merged_notes = np.append(merged_notes,[[pitch, onset_time, duration]],axis=0)
                        wild_vibrato = np.append(wild_vibrato,[[pitch, onset_time, duration]],axis=0)
                    else:
                        merged_notes = np.append(merged_notes,note[onset_note:offset_note+1,:],axis=0)
                    note[onset_note:offset_note+1,0] = 0
                else:
                    merged_notes = np.append(merged_notes,[note[n,:]],axis=0)
            elif note[n,0]!=0 and n+1>note.shape[0]-1:
                merged_notes = np.append(merged_notes,[note[-1,:]],axis=0)
        # append last note 
        return merged_notes, wild_vibrato

class LongSlide(Common):

    def __init__(self, melody, hop=256, sr=44100, max_transition_note_duration=0.09, min_transition_note_duration=0.015):
        """
        Creates a new Wav object instance of the given file.

        :param filename: name of the .wav file

        """
        self.melody = melody
        self.technique = 'slide-out'
        self.hop = hop
        self.sr = sr
        self.max_transition_note_duration = max_transition_note_duration
        self.min_transition_note_duration = min_transition_note_duration
        self.long_slide_sec = None
        self.quantised_melody = LongSlide.quantize(self.melody)
        
    @staticmethod
    def quantize(data, partitions=range(0, 90, 1), codebook=range(0, 91, 1)):
        """
        Quantise array into given scale.

        Usage:
          index, quants = quantize([3, 34, 84, 40, 23], range(10, 90, 10), range(10, 100, 10))
          >>> index
          [0, 3, 8, 3, 2]
          >>> quants
          [10, 40, 90, 40, 30]
          
        """
        indices = []
        quantised_data = []
        halfstep = float(partitions[1]-partitions[0])/2
        for datum in data:
            index = 0
            while index < len(partitions) and datum >= partitions[index]-halfstep:
                index += 1
            indices.append(index-1)
            quantised_data.append(codebook[index-1])
        indices = np.asarray(indices)
        quantised_data = np.asarray(quantised_data)
        quantised_data[np.nonzero(quantised_data<0)[0]] = 0
        
        return quantised_data

    @staticmethod
    def frame2note(quantised_melody,hop,sr):
        """
        Convert pitch sequence into note[onset pitch duration]
        :param quantised_melody: quantised pitch sequence.
        :param hop:              the hop size of pitch contour.
        :param sr:               the sampling rate of pitch contour.
        :returns:                note [onset pitch duration]
        """
        note = np.empty([0,3])
        frame = quantised_melody.copy()
        for f in range(frame.shape[0]-1):
            # The frame is not polyphonic and the frame is voiced 
            if frame[f]!=0:
                pitch = frame[f]
                onset = f
                offset = f
                while(frame[offset+1]==frame[offset] and offset+1<frame.shape[0]):
                    offset = offset+1
                duration = offset-onset+1
                note = np.append(note,[[pitch,onset,duration]],axis=0)
                frame[onset:offset+1] = 0
        note[:,1] = note[:,1]*hop/sr
        note[:,2] = note[:,2]*hop/sr
        return note

    def detect(self, expression_style_note):
        """
        Find long stair pattern(distance greater than three semitones) in quantised pitch sequence.
        :param pitch_contour:                quantised pitch sequence.
        :param hop:                          the step size of melody contour.
        :param sr:                           the sampling rate of melody contour.
        :param max_transition_note_duration: the maximal lenght of the note in middle of the ladder.
        :param min_transition_note_duration: the minimal lenght of the note in middle of the ladder.

        """
        # find downward-long-stairs
        # convert frame-level pitch contour into notes
        self.long_slide_sec = np.empty([0,2])
        note = LongSlide.frame2note(self.quantised_melody, self.hop, self.sr)
        for n in range(note.shape[0]-1):
            if note[n,0]!=0:
                pitch = note[n,0]
                onset_note = n
                offset_note = n
                # trace the ladder pattern
                while(note[offset_note+1,0]+1==note[offset_note,0] and \
                    note[offset_note+1,2]>=self.min_transition_note_duration and \
                    note[offset_note+1,2]<=self.max_transition_note_duration and \
                    offset_note+2<note.shape[0]):
                    offset_note = offset_note+1
                step = offset_note-onset_note+1
                # recognized as long slide if the step number of ladder is larger than 5
                if step>=5:
                    onset_time = note[onset_note,1]
                    offset_time = note[offset_note,1]+note[offset_note,2]
                    self.long_slide_sec = np.append(self.long_slide_sec,[[onset_time,offset_time]],axis=0)
                note[onset_note:offset_note+1,0] = 0

        # convert time segment of slide-out into note event
        self.long_slide = self.long_slide_sec_2_long_slide(self.long_slide_sec, expression_style_note[:,0:3])
        # update expression_style_note array
        expression_style_note = Common.update(expression_style_note=expression_style_note, 
                                              note_with_expression_style=self.long_slide, 
                                              technique=self.technique, 
                                              sub_technique=1)
        return expression_style_note

    @staticmethod
    def long_slide_sec_2_long_slide(long_slide_sec, note):
        long_slide = np.empty([0,3])
        for r_lss in range(len(long_slide_sec)):
            for r_mn in range(len(note)):
                if long_slide_sec[r_lss,0]>note[r_mn,1] and long_slide_sec[r_lss,0]<note[r_mn,1]+note[r_mn,2]:
                    long_slide_pitch = note[r_mn,0]
                    long_slide_onset = note[r_mn,1]
                    if r_mn+1<=len(note):
                        # loop from the next note
                        for r_mn_r in range(r_mn+1,len(note)):
                            if note[r_mn_r,1]+note[r_mn_r,2]>=long_slide_sec[r_lss,1]:
                                if note[r_mn_r,1]>long_slide_sec[r_lss,1]:
                                    long_slide_offset = long_slide_sec[r_lss,1]
                                    long_slide_dur = long_slide_offset-long_slide_onset
                                    break
                                else:
                                    long_slide_offset = note[r_mn_r,1]+note[r_mn_r,2]
                                    long_slide_dur = long_slide_offset-long_slide_onset
                                    break
                    else:
                        long_slide_offset = long_slide_sec[r_lss,1]
                        long_slide_dur = long_slide_offset-long_slide_onset
                    long_slide_note = [long_slide_pitch, long_slide_onset, long_slide_dur]
                    long_slide = np.append(long_slide, [long_slide_note], axis=0)
        return long_slide

    def evaluate(self,answer_path): 

        if type(answer_path).__name__=='ndarray':
            answer = answer_path.copy()
        else:
            answer = np.loadtxt(answer_path)
        numTP = 0.
        TP = np.array([])
        FP = np.array([])
        FN = np.array([])
        estimation = self.long_slide_sec.copy()
        estimation_mask = np.ones(len(self.long_slide_sec))
        answer_mask = np.ones(len(answer))
        for e in range(len(estimation)):
            for a in range(len(answer)):    
                if answer[a,0]>=estimation[e,0] and answer[a,0]<=estimation[e,1]:
                    answer_mask[a] = 0
                    estimation_mask[e] = 0
                    numTP = numTP+1
        numFN = np.sum(answer_mask)
        numFP = np.sum(estimation_mask)
        TP = estimation[np.nonzero(estimation_mask==0)[0]]
        FP = estimation[np.nonzero(estimation_mask==1)[0]]
        FN = answer[np.nonzero(answer_mask==1)[0]]
        P = numTP/float(numTP+numFP)
        R = numTP/float(numTP+numFN)
        F = 2*P*R/float(P+R)

        report.write()
        
        return P, R, F, TP, FP, FN, numTP, numFP, numFN



class SlowBend(Common):
    """
    Detect slow bend by the following rules:
        i) 
        ii) 

    :param note:        np.ndarray, shape=(n_event, 3)
                        note event[pitch(MIDI), onset, duration]
    :param CAD_pattern: np.ndarray, shape=(n_event, 2)
                        continuous ascending/descending pattern [start, end]
    :return:
    """
    def __init__(self, ascending_pattern, descending_pattern):
        self.technique = 'bend'
        self.ascending_pattern = ascending_pattern
        self.descending_pattern = descending_pattern
        self.slow_bend = None

    def detect(self, expression_style_note): 
                
        # long_ascending_note, note_short_ascending, long_ascending_pattern, short_ascending_pattern = long_CAD_pattern_detection(expression_style_note[:,0:3], self.ascending_pattern)
        slow_bend_note, short_ascending_pattern = SlowBend.long_CAD_pattern_detection(expression_style_note[:,0:3], self.ascending_pattern)
        # long_descending_note, note_short_descending, long_descending_pattern, short_descending_pattern = long_CAD_pattern_detection(expression_style_note[:,0:3], self.descending_pattern)
        slow_release_note, short_descending_pattern = SlowBend.long_CAD_pattern_detection(expression_style_note[:,0:3], self.descending_pattern)

        # update
        expression_style_note = Common.update(expression_style_note, slow_bend_note, technique = self.technique, sub_technique = 3)
        expression_style_note = Common.update(expression_style_note, slow_release_note, technique = 'release', sub_technique = 3)       

        return expression_style_note, slow_bend_note, slow_release_note, short_ascending_pattern, short_descending_pattern
    @staticmethod
    def long_CAD_pattern_detection(note, CAD_pattern):
        """
        Candidate selection for bend and slide by rules.
        All the candidates must meet: 
            i) continuously ascending or descending pattern covers three note.
            ii) The pitch difference of the three covered notes is a semitone

        :param      note:               2-D ndarray[pitch(MIDI). onset(s). duration(s)] 
                                        notes after mergin vibrato.

        :param      CAD_pattern:        1-D ndarray[onset(s). offset(s).]                
                                        continuously ascending or descending pattern.

        :returns    CAD_pattern:        1-D ndarray[onset(s). offset(s).]                
                                        continuously ascending or descending pattern.

        :returns    note_of_long_CAD:   1-D ndarray[onset(s). offset(s).]                
                                        continuously ascending or descending pattern.
        """
        note_of_long_CAD = np.empty([0,3])
        long_CAD_index = []
        note_of_long_CAD_index = []
        pseudo_CAD = CAD_pattern.copy()
        pseudo_note = note.copy()
        # Loop in each pattern
        for p in range(pseudo_CAD.shape[0]):
            onset_pattern = pseudo_CAD[p,0]
            offset_pattern = pseudo_CAD[p,1]
            # Loop in each note
            for n in range(pseudo_note.shape[0]):
                onset_note = pseudo_note[n,1]
                offset_note = pseudo_note[n,1]+pseudo_note[n,2]
                # Find notes where pattern located
                if onset_pattern >= onset_note and onset_pattern <= offset_note:
                    if n+3>=pseudo_note.shape[0]:
                        break
                    for m in range(n+2,n+4):
                        onset_note = pseudo_note[m,1]
                        offset_note = pseudo_note[m,1]+pseudo_note[m,2]
                        if offset_pattern >= onset_note and offset_pattern <= offset_note:
                            if m-n>=2 and m-n<=3 and abs(pseudo_note[n,0]-pseudo_note[m,0])<=3:
                                pitch = pseudo_note[n,0]
                                onset = pseudo_note[n,1]
                                duration = pseudo_note[n,2]+pseudo_note[n+1,2]+pseudo_note[n+2,2]
                                note_of_long_CAD = np.append(note_of_long_CAD,[[pitch, onset, duration]],axis = 0)
                                long_CAD_index.append(p)
                                note_of_long_CAD_index.append(n)
                                note_of_long_CAD_index.append(n+1)
                                note_of_long_CAD_index.append(n+2)
        long_CAD = pseudo_CAD[long_CAD_index,:]
        short_CAD = np.delete(pseudo_CAD,long_CAD_index,axis=0)
        note_of_short_CAD = np.delete(pseudo_note,note_of_long_CAD_index,axis=0)
        # return note_of_long_CAD, note_of_short_CAD, long_CAD, short_CAD
        return note_of_long_CAD, short_CAD

def merge_and_update_prebend_bend_release(expression_style_note, result_ref):
    import sys
    save_stdout = sys.stdout
    fh = open('/Users/Frank/Documents/Code/Python/Test_ExpressionStyle_result/lick_80/S3.ExpressionStyle/debug/after_S.5.4_Merge_and_update_prebend_bend_release/merge_and_update_prebend_bend_release_procedure.txt','w')
    sys.stdout = fh

    result = result_ref.copy()
    note_to_be_deleted = np.empty([0])
    for index_candi, candi_result in enumerate(result):
        # if the candidate is classsified as bend
        if candi_result[-1] == 0 and candi_result[0] != 0:
            for index_note, note in enumerate(expression_style_note[:-1]):
                # if the candidate exact covers consecutive two notes:
                if candi_result[0] > note[1] and candi_result[0] < note[1]+note[2] and \
                   candi_result[1] > expression_style_note[index_note+1,1] and \
                   candi_result[1] < expression_style_note[index_note+1,1]+expression_style_note[index_note+1,2]:
                    current_index_candi = index_candi
                    current_index_note = index_note+1
                    print 'Candidate exact covers consecutive two notes'
                    print '    index_candi: ', index_candi
                    print '    current_index_candi: ', current_index_candi
                    print '    index_note: ', index_note
                    print '    current_index_note: ', current_index_note
                    while current_index_note+1 <= expression_style_note.shape[0] and \
                          current_index_candi+1 <= result.shape[0] and \
                          result[current_index_candi+1,2] == 0 and \
                          result[current_index_candi+1,0] > expression_style_note[current_index_note,1] and \
                          result[current_index_candi+1,0] < expression_style_note[current_index_note,1]+expression_style_note[current_index_note,2] and \
                          result[current_index_candi+1,1] > expression_style_note[current_index_note+1,1] and \
                          result[current_index_candi+1,1] < expression_style_note[current_index_note+1,1]+expression_style_note[current_index_note,2]:
                        current_index_candi+=1
                        current_index_note+=1
                    print '    Checking whether or not the followed candidate is predicted as bend.' 
                    print '    index_candi: ', index_candi
                    print '    current_index_candi: ', current_index_candi
                    print '    index_note: ', index_note
                    print '    current_index_note: ', current_index_note
                    # delete the note which is about to be merged
                    note_to_be_deleted = np.append(note_to_be_deleted,range(index_note+1,current_index_note+1), axis=0)
                    print '    note_to_be_deleted: ', note_to_be_deleted
                    # mark the merged candidate as 0
                    # if current_index_candi-index_candi > 0:
                    result[index_candi:current_index_candi+1, 0:2] = 0
                    # replace the duration of first note with the difference of the 2nd note offset and 1st note onset
                    expression_style_note[index_note,2]=expression_style_note[current_index_note,1]+expression_style_note[current_index_note,2]-expression_style_note[index_note,1]
                    # keep the predicted expression styles on merged notes which is going to be deleted
                    expression_style_note[index_note,6:]=np.nanmax(expression_style_note[index_note+1:current_index_note+1,6:], axis=0)
                    # mark the bend in expression style note                    
                    print '    range(index_note,current_index_note): ', range(index_note,current_index_note)
                    for n in range(index_note,current_index_note):
                        print '    pitch of expression_style_note[n,0]', expression_style_note[n,0]
                    for n in range(index_note,current_index_note):
                        print 'expression_style_note[n+1,0]: ', expression_style_note[n+1,0]
                        print 'expression_style_note[n,0]: ', expression_style_note[n,0]
                        pitch_diff = expression_style_note[n+1,0]-expression_style_note[n,0]
                        print '    pitch_diff ', pitch_diff
                        if pitch_diff > 0:
                            expression_style_note[index_note,4] = pitch_diff
                        elif pitch_diff < 0:
                            expression_style_note[index_note,5] = abs(pitch_diff)
                    print '    Note index: ', index_note
                    print '         Pitch: ', expression_style_note[index_note, 0]
                    print '         Bend: ', expression_style_note[index_note, 4]
                    print '         Release: ', expression_style_note[index_note, 5]
                    if expression_style_note[index_note, 4]==0 and expression_style_note[index_note, 5]!=0:
                        print '    Here should be a pre-bend.'
                        expression_style_note[index_note, 3] = expression_style_note[index_note, 5]
                    # replace the pitch of first note with the lowest pitch among index_note to current_index_note
                    expression_style_note[index_note,0]= np.min(expression_style_note[index_note:current_index_note+1,0])
    # print note_to_be_deleted
    expression_style_note = np.delete(expression_style_note, note_to_be_deleted, axis=0)

    sys.stdout = save_stdout
    fh.close()

    return expression_style_note

def update_pull_hamm_slide(expression_style_note, result_ref, tech_index_dic):
    if tech_index_dic.has_key('pull'):
        if type(tech_index_dic['pull']) is int:
            pull_index_list=[tech_index_dic['pull']]
        else:
            pull_index_list=tech_index_dic['pull']
    else: 
        pull_index_list=[]
    if tech_index_dic.has_key('hamm'):
        if type(tech_index_dic['hamm']) is int:
            hamm_index_list=[tech_index_dic['hamm']]
        else:
            hamm_index_list=tech_index_dic['hamm']
    else: 
        hamm_index_list=[]
    if tech_index_dic.has_key('slide'):
        if type(tech_index_dic['slide']) is int:
            slide_index_list=[tech_index_dic['slide']]
        else:
            slide_index_list=tech_index_dic['slide']
    else: 
        slide_index_list=[]

    target_tech_index_list = pull_index_list+hamm_index_list+slide_index_list
    
    result = result_ref.copy()
    for index_candi, candi_result in enumerate(result):
        # if the candidate is classified as target techniques
        if candi_result[-1] in target_tech_index_list and candi_result[0] != 0:
            for index_note, note in enumerate(expression_style_note[:-1]):
                # if the candidate exact covers consecutive two notes:
                if candi_result[0] > note[1] and candi_result[0] < note[1]+note[2] and \
                   candi_result[1] > expression_style_note[index_note+1,1] and \
                   candi_result[1] < expression_style_note[index_note+1,1]+expression_style_note[index_note+1,2]:
                    t = [k for k, v in tech_index_dic.iteritems() if v == candi_result[-1]][0]   
                    # pull
                    if t=='pull':
                        if expression_style_note[index_note, 6]==0 and \
                           expression_style_note[index_note+1, 6]==0:
                            expression_style_note[index_note, 6]=1
                            expression_style_note[index_note+1, 6]=2
                        elif expression_style_note[index_note, 6]!=0 and \
                           expression_style_note[index_note+1, 6]==0:
                            expression_style_note[index_note+1, 6]=2
                    # hamm
                    elif t=='hamm':
                        if expression_style_note[index_note, 7]==0 and \
                           expression_style_note[index_note+1, 7]==0:
                            expression_style_note[index_note, 7]=1
                            expression_style_note[index_note+1, 7]=2
                        elif expression_style_note[index_note, 7]!=0 and \
                           expression_style_note[index_note+1, 7]==0:
                            expression_style_note[index_note+1, 7]=2
                    # slide
                    elif t=='slide':
                        expression_style_note[index_note, 8]=1
                        expression_style_note[index_note+1, 8]=2

    return expression_style_note

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


def transition_locater(note,step,direction = None,min_note_duration = 0.05,gap_tolerence = 0.05):
    """
    Find the timestamp of transition of two consecutive notes.
    Usage:
    :param note:              array of notes [pitch(hertz) onset(sec) duration(sec)].
    :param step:              the pithc distance between two notes.
    :param min_note_duration: the minimal duration of two consecutive notes
    :param gap_tolerence:     the minimal silent gap between two notes

    """
    transition = []
    for n in range(note.shape[0]-1):
        # print 'abs(np.log(note[n,0])-np.log(note[n+1,0])) is ', abs(np.log(note[n,0])-np.log(note[n+1,0]))
        # print 'note[n,2] is ', note[n,2]
        # print 'note[n+1,2] is ', note[n+1,2]
        # print 'min_note_duration*fs/hop is ', min_note_duration*fs/hop
        # print abs(np.log(note[n,0])-np.log(note[n+1,0]))
        # print '(note[n+1,1]-(note[n,1]+note[n,2])) is ', (note[n+1,1]-(note[n,1]+note[n,2]))
        # print 'gap_tolerence*fs/hop is ', gap_tolerence*fs/hop
                                # the pitch difference of two consecutive notes must smaller than or equal to given step
        is_transition_candi =   (abs(np.log(note[n,0])-np.log(note[n+1,0])) < step*0.0578+0.0578/2 and \
                                # the pitch difference of two consecutive notes must larger than or equal to given step
                                abs(np.log(note[n,0])-np.log(note[n+1,0])) > step*0.0578-0.0578/2 and \
                                # the duration of first note must longer than threshold
                                note[n,2]>min_note_duration and \
                                # the duration of second note must longer than threshold
                                note[n+1,2]>min_note_duration and \
                                # the gap between two consecutive notes must smaller than threshold
                                (note[n+1,1]-(note[n,1]+note[n,2])) < gap_tolerence)
        if is_transition_candi:
            if direction == None:
                transition.append(np.mean([note[n,1]+note[n,2], note[n+1,1]]))
            elif direction == 'upward':
                if note[n,0]<note[n+1,0]:
                    transition.append(np.mean([note[n,1]+note[n,2], note[n+1,1]]))
            elif direction == 'downward':
                if note[n,0]>note[n+1,0]:
                    transition.append(np.mean([note[n,1]+note[n,2], note[n+1,1]]))


    transition = np.asarray(transition)
    return transition


def save_esn_for_visualization(esn, output_dir, name):    
    np.savetxt(output_dir+os.sep+name+'.preb.esn', esn[:,[0,1,2,3]], fmt='%s')
    np.savetxt(output_dir+os.sep+name+'.b.esn', esn[:,[0,1,2,4]], fmt='%s')
    np.savetxt(output_dir+os.sep+name+'.r.esn', esn[:,[0,1,2,5]], fmt='%s')
    np.savetxt(output_dir+os.sep+name+'.p.esn', esn[:,[0,1,2,6]], fmt='%s')
    np.savetxt(output_dir+os.sep+name+'.h.esn', esn[:,[0,1,2,7]], fmt='%s')
    np.savetxt(output_dir+os.sep+name+'.s.esn', esn[:,[0,1,2,8]], fmt='%s')
    np.savetxt(output_dir+os.sep+name+'.si.esn', esn[:,[0,1,2,9]], fmt='%s')
    np.savetxt(output_dir+os.sep+name+'.so.esn', esn[:,[0,1,2,10]], fmt='%s')
    np.savetxt(output_dir+os.sep+name+'.v.esn', esn[:,[0,1,2,11]], fmt='%s')
    np.savetxt(output_dir+os.sep+name+'.index.esn', np.hstack([esn[:,0:3], np.arange(esn.shape[0]).reshape(esn.shape[0],1)]), fmt='%s')


def save_cls_result_for_visualization(result_all, output_dir, name, tech_index_dic):
    answer_tech_dic = {'bend':[3,4,5], 'pull':[6], 'hamm':[7], 'slide':[8,9,10], 'vibrato':[11]}    
    target_tech_list = [t for t in tech_index_dic if t in answer_tech_dic.keys()]
    for t in target_tech_list:
        if t=='bend':
            np.savetxt(output_dir+os.sep+name+'.b.cls_result', result_all[np.where(result_all[:,2]==tech_index_dic[t])[0],:], fmt='%s')
        elif t=='pull':
            np.savetxt(output_dir+os.sep+name+'.p.cls_result', result_all[np.where(result_all[:,2]==tech_index_dic[t])[0],:], fmt='%s')
        elif t=='hamm':
            np.savetxt(output_dir+os.sep+name+'.h.cls_result', result_all[np.where(result_all[:,2]==tech_index_dic[t])[0],:], fmt='%s')            
        elif t=='slide':
            np.savetxt(output_dir+os.sep+name+'.s.cls_result', result_all[np.where(result_all[:,2]==tech_index_dic[t])[0],:], fmt='%s')            
        elif t=='vibrato':
            np.savetxt(output_dir+os.sep+name+'.v.cls_result', result_all[np.where(result_all[:,2]==tech_index_dic[t])[0],:], fmt='%s')            
            

def parse_input_files(input_files, ext):
    """
    Collect all files by given extension.

    :param input_files:  list of input files or directories.
    :param ext:          the string of file extension.
    :returns:            a list of stings of file name.

    """
    from os.path import basename, isdir
    import fnmatch
    import glob
    files = []

    # check what we have (file/path)
    if isdir(input_files):
        # use all files with .raw.melody in the given path
        files = fnmatch.filter(glob.glob(input_files+'/*'), '*'+ext)
    else:
        # file was given, append to list
        if basename(input_files).find(ext)!=-1:
            files.append(input_files)
    print '  Input files: '
    for f in files: print '    ', f
    return files

def parser():
    """
    Parses the command line arguments.

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
    p.add_argument('input_audios', type=str, metavar='input_audios',
                   help='audio files to be processed')

    p.add_argument('input_melody', type=str, metavar='input_melody',
                   help='melody contours to be processed')

    p.add_argument('input_note', type=str, metavar='input_note',
                   help='note events to be processed')

    p.add_argument('input_model', type=str, metavar='input_model',
                   help='pre-trained classifier')

    p.add_argument('output_dir', type=str, metavar='output_dir',
                   help='output directory.')

    p.add_argument('-p',   '--prunning_note', dest='p',  
                   help="the minimum duration of note event.",  default=0.1)
    # debug
    p.add_argument('-debug', dest='debug', default=None, action='store_true',
                    help='result data to file for debugging.')
    # classification evaluation
    p.add_argument('-eval_cls', '--evaluation_classification', type=str, default=None, dest='eval_cls', 
                    help='Conduct classfication evaluation. The followed argument is parent directory of time-stamp annotation.')
    # expression style evaluation
    eval_expr = p.add_argument_group('Expression style recognition evaluation arguments')
    eval_expr.add_argument('-eval_expr', '--evaluation_expression_style', type=str, default=None, dest='eval_expr', 
                    help='Conduct note evaluation. The followed argument is parent directory of annotation.')
    # note evaluation
    eval_note = p.add_argument_group('Note evulation arguments')
    eval_note.add_argument('-eval_note', '--evaluation_note', type=str, default=None, dest='eval_note', 
                    help='Conduct note evaluation. The followed argument is parent directory of annotation.')

    eval_note.add_argument('-onset_tol', '--onset_tolerance_window', type=float, dest='onset_tol', default=0.05, 
                    help='Window lenght of onset tolerance. (default: %(default)s)')

    eval_note.add_argument('-offset_rat', '--offset_tolerance_ratio', type=float, dest='offset_rat', default=0.2, 
                    help='Window lenght of onset tolerance. (default: %(default)s)')
    # print
    p.add_argument('-v', dest='verbose', default=None, action='store_true',
                    help='be verbose')
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2016-05-04)')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return args
    return args


def main(args):
    print '======================================='
    print 'Running expression style recognition...'
    print '======================================='
    # parse and list files to be processed
    audio_files = parse_input_files(args.input_audios, ext='.wav')
    
    # create result directory
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    print '  Output directory: ', '\n', '    ', args.output_dir

    if args.debug:
        if not os.path.exists(args.output_dir+os.sep+'debug'): os.makedirs(args.output_dir+os.sep+'debug')

    for f in audio_files:
        ext = os.path.basename(f).split('.')[-1]
        name = os.path.basename(f).split('.')[0]      
        # load melody 
        melody_path = args.input_melody+os.sep+name+'.MIDI.smooth.melody'
        try:
            MIDI_smooth_melody = np.loadtxt(melody_path)
        except IOError:
            print 'The melody contour of ', name, ' doesn\'t exist!'

        # load raw note
        note_path = args.input_note+os.sep+name+'.raw.note'
        try:
            raw_note = np.loadtxt(note_path)
        except IOError:
            print 'The note event of ', name, ' doesn\'t exist!'  


        if args.eval_expr:
            annotation_note = np.loadtxt(args.eval_note+os.sep+name+'.note.answer')
            evaluation_note(annotation_note, raw_note, args.output_dir, name, onset_tolerance=args.onset_tol, offset_ratio=args.offset_rat, mode='w')

        """
        =====================================================================================
        S.1 Detect {wild vibrato} by recognizing the serrated pattern in note events.
        =====================================================================================
        """
        
        WV = WildVibrato()
        expression_style_note = WV.detect(raw_note)

        if args.debug:
            # create result directory
            debug_dir = args.output_dir+os.sep+'debug'+os.sep+'after_S.1_Wild_vibrato_detection'
            if not os.path.exists(debug_dir): 
                    os.makedirs(debug_dir)
            # save expression_style_note
            np.savetxt(debug_dir+os.sep+name+'.super_wild_vibrato',WV.super_wild_vibrato, fmt='%s')
            np.savetxt(debug_dir+os.sep+name+'.wild_vibrato',WV.wild_vibrato, fmt='%s')
            np.savetxt(debug_dir+os.sep+name+'.esn', expression_style_note, fmt='%s')
            save_esn_for_visualization(expression_style_note, debug_dir, name)

        if args.eval_expr:
            print '  Evaluating expression style note after wild vibrato detection...' 
            annotation_esn = np.loadtxt(args.eval_expr+os.sep+name+'.esn.answer')
            evaluation_expr(annotation_esn, expression_style_note, args.output_dir, name, onset_tolerance=0.05, offset_ratio=0.2, 
                string='Result after wild vibrato detection', mode='w')


        """
        =====================================================================================
        S.2 Detect {slide in} {slide out} by recognizing the ladder pattern in quantised melody contour.
        =====================================================================================
        """
        LS = LongSlide(MIDI_smooth_melody, hop=contour_hop, sr=contour_sr, 
                       max_transition_note_duration=max_transition_note_duration, 
                       min_transition_note_duration=min_transition_note_duration)
        expression_style_note = LS.detect(expression_style_note)        

        if args.debug:
            # create result directory
            debug_dir = args.output_dir+os.sep+'debug'+os.sep+'after_S.2_Slide_in_slide_out_detection'
            if not os.path.exists(debug_dir): 
                    os.makedirs(debug_dir)
            # save updated expression style note
            np.savetxt(debug_dir+os.sep+name+'.esn', expression_style_note, fmt='%s')
            save_esn_for_visualization(expression_style_note, debug_dir, name)
            # save outputs
            np.savetxt(debug_dir+os.sep+name+'.quantised.melody',LS.quantised_melody, fmt='%s')
            np.savetxt(debug_dir+os.sep+name+'.long_slide',LS.long_slide, fmt='%s')

        if args.eval_expr:
            print '  Evaluating expression style note after slide in/out detection...' 
            annotation_esn = np.loadtxt(args.eval_expr+os.sep+name+'.esn.answer')
            evaluation_expr(annotation_esn, expression_style_note, args.output_dir, name, onset_tolerance=0.05, offset_ratio=0.2, 
                string='Result after slide in/out detection', mode='a')

        """
        =====================================================================================
        S.3 Find continuously ascending or descending (CAD) pattern in melody contour.
        =====================================================================================
        """
        # find continuously ascending (CAD) F0 sequence patterns
        ascending_pattern, ascending_pitch_contour = continuously_ascending_descending_pattern(
                                MIDI_smooth_melody,direction='up',MinLastingDuration=0.05, 
                                MaxPitchDifference=3.8, MinPitchDifference=0.8,hop=contour_hop,sr=contour_sr)
        # find continuously descending (CAD) F0 sequence patterns
        descending_pattern, descending_pitch_contour = continuously_ascending_descending_pattern(
                                MIDI_smooth_melody,direction='down',MinLastingDuration=0.05, 
                                MaxPitchDifference=3.8, MinPitchDifference=0.8,hop=contour_hop,sr=contour_sr)
        # save result: CAD F0 sequence pattern
        np.savetxt(args.output_dir+os.sep+name+'.ascending.pattern',ascending_pattern, fmt='%s')
        np.savetxt(args.output_dir+os.sep+name+'.ascending.pitch_contour',ascending_pitch_contour, fmt='%s')
        np.savetxt(args.output_dir+os.sep+name+'.descending.pattern',descending_pattern, fmt='%s')
        np.savetxt(args.output_dir+os.sep+name+'.descending.pitch_contour',descending_pitch_contour, fmt='%s')

        """    
        =====================================================================================            
        S.4 Detect {slow bend} by using following heuristic rule: 
            i) search three or four consecutive adjacent notes differed by a semitone which covered by CAD pattern.
            ii)
        =====================================================================================
        """

        SB = SlowBend(ascending_pattern, descending_pattern)
        (expression_style_note, slow_bend_note, slow_release_note, 
        short_ascending_pattern, short_descending_pattern) = SB.detect(expression_style_note) 

        if args.debug:
            # create result directory
            debug_dir = args.output_dir+os.sep+'debug'+os.sep+'after_S.4_Slow_bend_release_detection'
            if not os.path.exists(debug_dir): 
                    os.makedirs(debug_dir)
            # save updated expression style note
            np.savetxt(debug_dir+os.sep+name+'.esn', expression_style_note, fmt='%s')
            save_esn_for_visualization(expression_style_note, debug_dir, name)
            # save slow bend and slow release
            np.savetxt(debug_dir+os.sep+name+'.slow_bend',slow_bend_note, fmt='%s')
            np.savetxt(debug_dir+os.sep+name+'.slow_release',slow_release_note, fmt='%s')
           
        if args.eval_expr:
            print '  Evaluating expression style note after slow bend detection...' 
            annotation_esn = np.loadtxt(args.eval_expr+os.sep+name+'.esn.answer')
            evaluation_expr(annotation_esn, expression_style_note, args.output_dir, name, onset_tolerance=0.05, offset_ratio=0.2, 
                string='Result after slow bend detection', mode='a')

        """
        ===================================================================================================
        S.5 Detect {bend} {hammer on} {pull off} {slide} by analyzing timbre of selected candidate regions.
        ===================================================================================================
        """

        """
        -----------------------------------------------------------------------------
        S.5.1 Candidate selection by finding note transitions covered by CAD pattern.
              i.e., the candidate of {bend, hammer-on, normal, pull-off, slide}.
        -----------------------------------------------------------------------------
        """                          
        
        ascending_candidate, ascending_candidate_note, non_candidate_ascending_note = candidate_selection(expression_style_note[:,0:3], short_ascending_pattern)
        # num_valid_candidate, num_invalid_candidate, invalid_candidate, TP_bend, TP_slide, TP_hamm, FN_bend, FN_slide, FN_hamm = short_pattern_evaluate(ascending_candidate,FN_bend,FN_slide,join(hamm_answer_dir,name_ext))   
        descending_candidate, descending_candidate_note, non_candidate_descending_note = candidate_selection(expression_style_note[:,0:3], short_descending_pattern)
        # num_valid_candidate, num_invalid_candidate, invalid_candidate, TP_release, TP_slide, TP_pull, FN_release, FN_slide, FN_pull = short_pattern_evaluate(descending_candidate,FN_release,FN_slide,join(pull_answer_dir,name_ext))
        # save result: candidate
        np.savetxt(args.output_dir+os.sep+name+'.ascending.candidate',ascending_candidate, fmt='%s')
        np.savetxt(args.output_dir+os.sep+name+'.descending.candidate',descending_candidate, fmt='%s')
        
        """
        -----------------------------------------------------
        S.5.2 Extract features of selected candidate regions.
        -----------------------------------------------------
        """
        # load audio
        audio = MonoLoader(filename = f)()
        # processing
        print '     Processing file: ', f
        candidate_type = ['ascending','descending']
        # loop in ascending and descending candidate list
        for ct in candidate_type:
            print '         Extracting features from', ct, 'candidates...'
            # candidate file path
            candidate_path = args.output_dir+os.sep+name+'.'+ct+'.candidate'
            # inspect if candidate file exist and load it
            try:
                candidate = np.loadtxt(candidate_path)
            except IOError:
                print 'The candidate of', name,' doesn\'t exist!'
            # reshape candidate if it is in one dimension
            if candidate.shape==(2,): candidate = candidate.reshape(1,2)
            # convert seconds into samples
            candidate_sample = candidate*contour_sr
            # create feature matrix
            feature_vec_all = np.array([])
            # loop in candidates
            for c in candidate_sample:
                # clipping audio signal
                audio_clip = audio[int(c[0]):int(c[1])]
                # extract features
                feature_vec = feature_extractor(audio=audio_clip, features=selected_features)
                feature_vec_all = np.concatenate((feature_vec_all,feature_vec), axis = 0)            
            # reshpe feature vector and save result
            if feature_vec_all.size!=0:
                feature_vec_all = feature_vec_all.reshape(len(candidate_sample),len(feature_vec_all)/len(candidate_sample))
                np.savetxt(args.output_dir+os.sep+name+'.'+ct+'.candidate'+'.raw.feature', feature_vec_all, fmt='%s')


        """
        -----------------------------------------------
        S.5.3 Classfication using pre-train classifier.
        -----------------------------------------------
        """        
        # load pre-trained SVM
        tech_index_dic = {'bend':0, 'hamm':1, 'normal':2, 'pull':3, 'slide':4}
        try:
            clf = np.load(args.input_model).item()
        except IOError:
            print 'The expression style recognition model ', args.input_model, ' doesn\'t exist!'

        for ct in candidate_type:
            # load raw features
            candidate = np.loadtxt(args.output_dir+os.sep+name+'.'+ct+'.candidate')
            raw_feature = np.loadtxt(args.output_dir+os.sep+name+'.'+ct+'.candidate'+'.raw.feature')
            # raw_data = np.vstack((ascending_raw_feature, descending_raw_feature))

            # data preprocessing
            data = data_preprocessing(raw_feature)

            # classfication
            y_pred = clf.predict(data)
            result = np.hstack((candidate, np.asarray(y_pred).reshape(len(y_pred), 1)))
            np.savetxt(args.output_dir+os.sep+name+'.'+ct+'.candidate'+'.cls_result', result, fmt='%s')

        # combine ascending and descending cadidates
        result_all = np.vstack((np.loadtxt(args.output_dir+os.sep+name+'.'+candidate_type[0]+'.candidate'+'.cls_result'), np.loadtxt(args.output_dir+os.sep+name+'.'+candidate_type[1]+'.candidate'+'.cls_result')))
        # sort by time
        result_all = result_all[np.argsort(result_all[:,0], axis = 0)]

        if args.debug:
            # create result directory
            debug_dir = args.output_dir+os.sep+'debug'+os.sep+'after_S.5.3_classification'
            if not os.path.exists(debug_dir): 
                    os.makedirs(debug_dir)
            np.savetxt(debug_dir+os.sep+name+'.all.cls_result', result_all, fmt='%s')
            save_cls_result_for_visualization(result_all, debug_dir, name, tech_index_dic=tech_index_dic)

        if args.eval_cls:
            print '  Evaluating classification result...' 
            # load time-stamp answer
            annotation_ts = np.loadtxt(args.eval_cls+os.sep+name+'.ts.answer')
            evaluation_candidate_classification(annotation_ts, result_all, args.output_dir, name, 
                tech_index_dic=tech_index_dic, string=None, mode='w')

        """
        -------------------------------------------------------------------------------------------
        S.5.4 Merge bend & release notes and update expression style note by classification result.
        -------------------------------------------------------------------------------------------
        """
        
        # merge bend and note
        expression_style_note = merge_and_update_prebend_bend_release(expression_style_note, result_all)

        if args.debug:
            # create result directory
            debug_dir = args.output_dir+os.sep+'debug'+os.sep+'after_S.5.4_Merge_and_update_prebend_bend_release'
            if not os.path.exists(debug_dir): 
                    os.makedirs(debug_dir)
            # save updated expression style note
            np.savetxt(debug_dir+os.sep+name+'.esn', expression_style_note, fmt='%s')
            save_esn_for_visualization(expression_style_note, debug_dir, name)

        if args.eval_expr:
            print '  Evaluating expression style note after bended notes merged...' 
            # load esn answer
            annotation_esn = np.loadtxt(args.eval_expr+os.sep+name+'.esn.answer')
            evaluation_expr(annotation_esn, expression_style_note, args.output_dir, name, onset_tolerance=0.05, offset_ratio=0.2, 
                string='Result after bended notes merged.', mode='a')

        """
        ---------------------------------------------------------------------------------------------------
        S.5.5 Update pull-off, hammer-on and slide notes to expression style note by classification result.
        ---------------------------------------------------------------------------------------------------
        """

        # update esn
        expression_style_note = update_pull_hamm_slide(expression_style_note, result_all, tech_index_dic=tech_index_dic)

        if args.debug:
            # create result directory
            debug_dir = args.output_dir+os.sep+'debug'+os.sep+'after_S.5.5_Update_pull_hamm_slide'
            if not os.path.exists(debug_dir): 
                    os.makedirs(debug_dir)
            # save updated expression style note
            np.savetxt(debug_dir+os.sep+name+'.esn', expression_style_note, fmt='%s')
            save_esn_for_visualization(expression_style_note, debug_dir, name)

        if args.eval_expr:
            print '  Evaluating expression style note after pull-off, hammer-on and slide updated with classification result...' 
            # load esn answer
            annotation_esn = np.loadtxt(args.eval_expr+os.sep+name+'.esn.answer')
            evaluation_expr(annotation_esn, expression_style_note, args.output_dir, name, onset_tolerance=0.05, offset_ratio=0.2, 
                string='Result after pull-off, hammer-on and slide notes updated.', mode='a')

        if args.eval_note:
            print '  Evaluating note accuracy...'
            # load note answer
            annotation = np.loadtxt(args.eval_note+os.sep+name+'.note.answer')
            note = expression_style_note[:,0:3]
            # pruned_note = note_pruning(note, threshold=args.p)
            evaluation_note(annotation, note, args.output_dir, name, onset_tolerance=args.onset_tol, offset_ratio=args.offset_rat, 
                string='Result after pull-off, hammer-on and slide notes updated.', mode='a')
            
        
        """
        ====================================================================================================================
        S.6 Detect {hammer on} {pull off} by analyzing the timbre of remained note transition which are not selected in S.5.
        ====================================================================================================================
        """

        """
        ==================================================================================
        S.7 Detect {vibrato} on each note.
        ==================================================================================
        """

        

if __name__ == '__main__':
    args = parser()
    main(args)
