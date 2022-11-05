from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
import operator
from . import parameters as pm
from .contour import *
from .technique import *
from .note import *
from scipy.stats import norm
from os import sep

#=====Parameters=====#
min_pitch=30.0
min_melo_len=18
min_pattern_length=8
min_vib_amp=0.3
min_cs_amp=0.8
max_cont_diff=0.8
max_cand_diff=3.5
max_cs_amp=3.0
max_cs_length=33

nf_weights = np.array([norm.pdf(i, scale=2) for i in range(-5, 6)])
nf_weights /= nf_weights.sum()

def conditioned_norm_filter(data):
    new_data = np.zeros(data.shape)
    h_fil = old_div(len(nf_weights), 2)
    for i in range(len(data)):
        if data[i] < min_pitch: continue
        v = np.array([ data[i-j] if ( 0 <= i-j < len(data) and \
                                      data[i-j] >= min_pitch and \
                                      np.abs(data[i-j] - data[i]) <= max_cont_diff \
                                    ) else 0
                       for j in range(-h_fil, h_fil + 1)])
        w_sum = np.extract(v != 0, nf_weights).sum()
        new_data[i] = old_div((v * nf_weights).sum(), w_sum)
    return new_data

def conditioned_mean_filter(data, filter_size=5):
    if filter_size % 2 == 0:
        filter_size += 1
        print('Filter size should be odd. Set filer size to {}.'.format(filter_size))
    new_data = np.zeros(data.shape)
    h_fil = old_div(filter_size, 2)
    for i in range(len(data)):
        if data[i] < min_pitch: continue
        v = np.array([data[i-j] for j in range(-h_fil, h_fil + 1)
                        if 0 <= i-j < len(data) and \
                        data[i-j] >= min_pitch and \
                        abs(data[i-j] - data[i]) <= 0.5
                     ], dtype=float)
        new_data[i] = round(v.mean(), 4)
    return new_data

### Technique Embedded Note Tracking
def tent(melody, debug=None):
    if melody.length == 0:
        print('Nothing in melody. (Length of melody is 0.)')
        return
    melody = Contour(melody.start_idx, 
                     conditioned_norm_filter(melody.seq)
                    )
    submelo_list = []
    sub_idx = 0
    submelo = []
    melody_cand_dict = {}
    for i in range(melody.length):
        if len(submelo) == 0:
            if melody[i] >= min_pitch:
                sub_idx = i
                submelo.append(melody[i])
        elif abs(melody[i] - submelo[-1]) <= max_cont_diff:
            submelo.append(melody[i])
        else:
            if len(submelo) >= min_melo_len:
                ct = Contour(sub_idx, submelo)
                if len(submelo_list) > 0 and \
                   submelo_list[-1].end_idx + 1 == sub_idx and \
                   abs(submelo[0] - submelo_list[-1][-1]) < max_cand_diff:
                    ### Select as Candidate
                    sign = 1 if submelo[0] >= submelo_list[-1][-1] else -1
                    melody_cand_dict[len(submelo_list)] = (sign, sub_idx)
                submelo_list.append(ct)
            submelo = []
            if melody[i] >= min_pitch:
                sub_idx = i
                submelo.append(melody[i])
    if len(submelo) >= min_melo_len:
        ct = Contour(sub_idx, submelo)
        submelo_list.append(ct)

    trend = np.zeros(melody.length)
    if debug is not None: mid_trend = np.zeros(melody.length)
    # n_melo = melody.sub_contour(range(melody.length))
    notes = []
    for idx, subm in enumerate(submelo_list):
        tr = melody_2_trend(subm)
        if debug is not None: mid_trend[subm.start_idx:subm.start_idx+len(tr)] = list(tr)
        nt = get_notes(subm, tr)
        ### Add candidate between submelodies
        if idx in list(melody_cand_dict.keys()):
            sign, sub_idx = melody_cand_dict[idx]
            seg_pos = max(0, sub_idx - old_div(pm.MC_LENGTH,2) - notes[-1].onset)
            seg = Segment(sign, seg_pos, pm.MC_LENGTH, melody)
            notes[-1].segs.append(seg)
            notes[-1].next_note = nt[0]

        trend[subm.start_idx:subm.start_idx+len(tr)] = tr
        # n_melo.seq[subm.start_idx:subm.start_idx+mm.length] = mm[:]
        notes += nt

    if debug is not None:
        np.savetxt(debug+sep+'MidTrend.txt', mid_trend)
    return trend, melody, notes

def melody_2_trend(melody):
    extrema = get_extrema(melody.seq)
    ### If the difference in this melody is smaller than min_vib_amp, 
    ### return a single, nontechnical note.
    if max(extrema[:,1]) - min(extrema[:,1]) < min_vib_amp:
        return [0] * melody.length

    ### Record the trend (ascending, descending, or horizontal)
    trend = np.zeros(melody.length)
    for i in range(len(extrema)-1):
        j, j_val, _ = extrema[i]
        k, k_val, _ = extrema[i+1]
        j, k = int(j), int(k)
        pattern = Contour(j, melody[j:k])
        trend[j:k] = scan_pattern_trend(pattern, melody[int(k)])
    trend[-1] = trend[-2] 
    return trend

def scan_pattern_trend(pattern, next_extreme, alpha=0.5):
    pattern_diff = next_extreme - pattern[0]
    trend = [0] * pattern.length
    ### Trace the pattern that is possible to find techs and highlight the part of slope
    if (abs(pattern_diff) >= min_vib_amp):
        ### Decide the direction of pattern
        if pattern_diff >= 0: ### ascending pattern
            trend_type = 1
            opt = operator.gt
        elif pattern_diff < 0: ### descending pattern
            trend_type = -1
            opt = operator.lt
        else:
            raise ValueError("Direction must either be \'up\' or \'down\'")

        ### Find place with slope larger than average slope
        slope = alpha * pattern_diff / pattern.length
        accu_plain = 0
        trend = [0] * pattern.length
        plain_thres = min(old_div(pattern.length,3), 18)
        m = end_m = start_m = 0
        while m < pattern.length-1:
            if opt(pattern[m+1] - pattern[m], slope):
                end_m = m + 1
                if accu_plain > 0: 
                    accu_plain = 0
            else:
                accu_plain += 1
                if accu_plain == plain_thres:
                    if end_m > start_m and \
                       abs(pattern[end_m] - pattern[start_m]) >= min_cs_amp:
                        trend[start_m:end_m] = [trend_type] * (end_m  - start_m)
                    start_m = end_m
                if end_m == start_m:
                    start_m = end_m = m + 1
            m += 1
        if accu_plain < plain_thres and abs(next_extreme - pattern[start_m]) >= min_vib_amp:
            end_m = pattern.length
            trend[start_m:end_m] = [trend_type] * (end_m  - start_m)
    return trend

def get_notes(melody, trend):
    ### Merge segments
    seg_melo = SegmentedContour(melody.start_idx, melody.seq, trend)
    if seg_melo.n_segs > 0:
        ### Fill some small zero holes in some trends
        all_segs = seg_melo.all_segs(sort=True)
        s = all_segs[0]
        for ns in all_segs[1:]:
            if ns.val == s.val and \
               ns.pos - s.end < min_pattern_length:
                seg_melo.merge_segs([s.pos, ns.pos])
            else:
                s = ns
        
        for seg in seg_melo.all_segs():
            ### Some trends that happened in the head or tail of a submelody
            ### should be considered from melody scale, not here.
            if seg.mid < 10 or seg_melo.length - seg.mid < 10:
                seg_melo.delete_seg(seg)
                continue

            ### Check special techniques
            ct = seg.contour()
            if ct.max - ct.min >= 3.5:
                if seg.pos < min_pattern_length:
                    seg.val *= T_SLIDE_IN
                elif seg_melo.length - seg.end < min_pattern_length:
                    seg.val *= T_SLIDE_OUT
                else:
                    seg.val *= T_SLIDE
            elif ct.length >= 30:
                seg.val *= T_BEND

    ### Split the trend if there are several possible notes in this trend.
    notes = []
    start_point = 0
    cands = []
    slide_in = -1
    slide_out = -1
    all_segs = seg_melo.all_segs(sort=True)
    for i in range(len(all_segs)):
        if abs(all_segs[i].val) == T_SLIDE_IN:
            slide_in = seg_melo.start_idx + all_segs[i].pos
        elif abs(all_segs[i].val) == T_SLIDE_OUT:
            slide_out = seg_melo.start_idx + all_segs[i].pos
        elif i < len(all_segs)-1 and \
             np.sign(all_segs[i].val) == np.sign(all_segs[i+1].val) and \
             abs(all_segs[i+1].val) not in (T_SLIDE_IN, T_SLIDE_OUT):
            cands.append(seg_melo.start_idx + all_segs[i].pos)
            contour = seg_melo.sub_contour(list(range(start_point, all_segs[i].end)))
            notes += estimate_notes(contour, cands, slide_in, slide_out)
            trend[start_point:all_segs[i].end] = contour.get_trend()
            start_point = all_segs[i].end
            cands = []
            slide_in = -1
        else:
            cands.append(seg_melo.start_idx + all_segs[i].pos)
    contour = seg_melo.sub_contour(list(range(start_point, seg_melo.length)))
    notes += estimate_notes(contour, cands, slide_in, slide_out)
    ### update the trend
    trend[start_point:seg_melo.length] = contour.get_trend()
    merge_notes(notes)
    return notes

def estimate_notes(melo, cands, slide_in, slide_out):
    cidx = lambda k: cands[k] - melo.start_idx
    if len(cands) == 0:
        return create_note_0(melo, slide_in, slide_out)
    elif len(cands) == 1:
        seg = melo.seg(cidx(0))
        if seg.diff() >= 0.8:
            return create_note_1(melo, seg, slide_in, slide_out)
        else:
            melo.delete_seg(seg)
            return create_note_0(melo, slide_in, slide_out)
    elif len(cands) == 2:
        flag = [1, 1]
        for i in range(2):
            seg = melo.seg(cidx(i))
            if seg.diff() < 0.8: 
                melo.delete_seg(seg)
                flag[i] = 0
        cands = np.extract(flag, cands)
        if len(cands) == 0:
            return create_note_0(melo, slide_in, slide_out)
        elif len(cands) == 1:
            return create_note_1(melo, melo.seg(cidx(0)), slide_in, slide_out)
        else:
            return create_note_2(melo, melo.seg(cidx(0)), melo.seg(cidx(1)), slide_in, slide_out)
    else:
        melo.merge_segs([cidx(i) for i in range(len(cands))])
        seg = melo.seg(cidx(0))
        seg.val *= T_VIBRATO
        return create_vibrato_note(melo, seg, slide_in, slide_out)

def merge_notes(notes):
    for i in range(len(notes)-1, 0, -1):
        nt = notes[i-1]
        ### Merge notes if there is bend or release
        if (nt.tech(T_BEND).value > 0 and \
            nt.offset == notes[i].onset and \
            nt.pitch + nt.tech(T_BEND).value == notes[i].pitch) or \
           (nt.tech(T_RELEASE).value > 0 and \
            nt.offset == notes[i].onset and \
            nt.pitch == notes[i].pitch):
            notes[i-1] = CandidateNote.merge(nt, notes[i])
            notes.remove(notes[i])
        elif nt.tech(T_SLIDE).value == 1:
            t_val = 3 if notes[i].tech(T_SLIDE).value in (1, 3) else 2
            notes[i].add_tech(Tech(T_SLIDE, t_val))
        elif nt.tech(T_HAMMER).value == 1:
            t_val = 3 if notes[i].tech(T_HAMMER).value in (1, 3) else 2
            notes[i].add_tech(Tech(T_HAMMER, t_val))
        elif nt.tech(T_PULL).value == 1:
            t_val = 3 if notes[i].tech(T_PULL).value in (1, 3) else 2
            notes[i].add_tech(Tech(T_PULL, t_val))

def has_slide_in(melo, slide_in):
    """
    Get the object of SLIDE_IN in the melody and its ending point.

    Parameters
    ----------
    melo: SegmentedContour, the detecting melody
    slide_in: int, the starting index of SLIDE_IN

    Returns
    -------
    tech_object: Tech, the object of this SLIDE_IN, returns None if cannot find SLIDE_IN
    end_point: int, the end point of this SLIDE_IN on the melody
    """
    st = 0
    if slide_in >= 0:
        s = melo.seg(slide_in - melo.start_idx)
        st = s.end
        return Tech(abs(s.val), 1 if s.val > 0 else 2), st
    return None, st

def has_slide_out(melo, slide_out):
    """
    Get the object of SLIDE_OUT in the melody and its starting point.

    Parameters
    ----------
    melo: SegmentedContour, the detecting melody
    slide_in: int, the starting index of SLIDE_OUT

    Returns
    -------
    tech_object: Tech, the object of this SLIDE_OUT, returns None if cannot find SLIDE_OUT
    start_point: int, the start point of this SLIDE_OUT on the melody
    """
    ed = melo.length
    if slide_out >= 0:
        s = melo.seg(slide_out - melo.start_idx)
        ed = s.pos
        return Tech(abs(s.val), 1 if s.val < 0 else 2), ed
    return None, ed

def create_note_0(melo, slide_in, slide_out):
    techs = []
    in_tech, st = has_slide_in(melo, slide_in)
    if in_tech is not None: techs.append(in_tech)
    out_tech, ed = has_slide_out(melo, slide_out)
    if out_tech is not None: techs.append(out_tech)
    pitch = melo.estimated_pitch(list(range(st, ed)))
    return [CandidateNote(pitch, melo.start_idx, melo.length, techs=techs)]

def create_note_1(melo, seg, slide_in, slide_out):
    notes = []
    techs = []
    in_tech, st = has_slide_in(melo, slide_in)
    if in_tech is not None: techs.append(in_tech)
    out_tech, ed = has_slide_out(melo, slide_out)

    ### First Note
    pitch = melo.estimated_pitch(list(range(st, seg.pos)))
    note = get_note_with_seg(pitch, melo.start_idx, seg.end, techs, seg)
    notes.append(note)
    ### Second Note
    if melo.length > seg.end:
        techs2 = []
        if out_tech is not None: techs2.append(out_tech)
        # if note.tech(T_SLIDE).value == 1: techs2.append(Tech(T_SLIDE, 2))
        pitch2 = melo.estimated_pitch(list(range(seg.end, ed)))
        note2 = CandidateNote(pitch2, melo.start_idx + seg.end, melo.length - seg.end, techs=techs2)
        if isinstance(note, CandidateNote):
            ### Adjust onset and offset of two notes to the middle of seg
            note.duration = seg.mid
            nn_offset = note2.offset
            note2.onset = note.offset
            note2.duration = nn_offset - note2.onset
            note.next_note = note2
        notes.append(note2)
    return notes
    # if abs(seg.val) == T_BEND:
    #     ct = seg.contour()
    #     tval = int(round(ct.max - ct.min))
    #     ### Only one note
    #     if seg.val > 0:
    #         ttype = T_BEND
    #     else:
    #         ttype = T_RELEASE
    #         pitch -= tval
    #         techs.append(Tech(T_PREBEND, tval))
    #     techs.append(Tech(ttype, tval))
    #     if out_tech is not None: techs.append(out_tech)
    #     return [CandidateNote(pitch, melo.start_idx, melo.length, techs=techs)]
    # elif abs(seg.val) == T_SLIDE:
    #     ### First Note
    #     techs.append(Tech(T_SLIDE, 1))
    #     note1 = CandidateNote(pitch, melo.start_idx, seg.end, techs=techs)
    # else:
    #     ### First Note
    #     note1 = CandidateNote(pitch, melo.start_idx, seg.end, techs=techs, segs=[seg])
        

def create_note_2(melo, seg1, seg2, slide_in, slide_out):
    notes = []
    techs, techs2 = [], []
    in_tech, st = has_slide_in(melo, slide_in)
    if in_tech is not None: techs.append(in_tech)
    out_tech, ed = has_slide_out(melo, slide_out)
    if in_tech is not None: techs3.append(out_tech)
    pitch1 = melo.estimated_pitch(list(range(st, seg1.pos)))
    note1 = get_note_with_seg(pitch1, melo.start_idx, seg1.end, techs, seg1)
    notes.append(note1)
    
    ### Second Note
    pitch2 = melo.estimated_pitch(list(range(seg1.end, seg2.pos)))
    new_seg2 = Segment(seg=seg2)
    new_seg2.pos -= seg1.end
    note2 = get_note_with_seg(pitch2, melo.start_idx + seg1.end, seg2.end - seg1.end, techs2, new_seg2)
    if isinstance(note1, CandidateNote):
        note1.duration = seg1.mid
        nn_offset = note2.offset
        note2.onset = note1.offset
        note2.duration = nn_offset - note2.onset
        note1.next_note = note2
    notes.append(note2)
    # if note2.tech(T_BEND).value > 0 or note2.tech(T_RELEASE).value > 0:
    #     note2.duration = melo.length - seg1.end
    #     if out_tech is not None: note2.add_tech(out_tech)
    # el
    if melo.length > seg2.end:
        ### Third Note
        techs3 = []
        if out_tech is not None: techs3.append(out_tech)
        # if note2.tech(T_SLIDE).value == 1: techs3.append(Tech(T_SLIDE, 2))
        pitch3 = melo.estimated_pitch(list(range(seg2.end, ed)))
        note3 = CandidateNote(pitch3, melo.start_idx + seg2.end, melo.length - seg2.end, techs=techs3)
        if isinstance(note2, CandidateNote):
            note2.duration = new_seg2.mid
            nn_offset = note3.offset
            note3.onset = note2.offset
            note3.duration = nn_offset - note3.onset
            note2.next_note = note3
        notes.append(note3)
    return notes

def create_vibrato_note(melo, seg, slide_in, slide_out):
    techs = []
    in_tech, st = has_slide_in(melo, slide_in)
    if in_tech is not None: techs.append(in_tech)
    out_tech, ed = has_slide_out(melo, slide_out)
    if out_tech is not None: techs.append(out_tech)
    tval = max(int(round(seg.diff())), 1)
    techs.append(Tech(T_VIBRATO, tval))
    pitch = melo.estimated_pitch(list(range(st, seg.pos)))
    return [CandidateNote(pitch, melo.start_idx, melo.length, techs=techs)]


def get_note_with_seg(pitch, start_idx, length, techs, seg):
    if abs(seg.val) == T_BEND:
        tval = int(round(seg.diff()))
        if seg.val > 0:
            ttype = T_BEND
        else:
            ttype = T_RELEASE
            pitch = int(round(seg.contour().min))
            techs.append(Tech(T_PREBEND, tval))
        techs.append(Tech(ttype, tval))
        return CandidateNote(pitch, start_idx, length, techs=techs)
    elif abs(seg.val) == T_SLIDE:
        techs.append(Tech(T_SLIDE, 1))
        return CandidateNote(pitch, start_idx, length, techs=techs)
    else:
        return CandidateNote(pitch, start_idx, length, techs=techs, segs=[seg])




def _generate_extrema(y, y_cond, etype):
    ### Extract extrema from pruned matrix and add the extrema type to the matrix
    e = y[np.extract(y_cond, list(range(len(y_cond))))]
    ### np.full is faster than np.vstack
    ext = np.full((e.shape[0], e.shape[-1]+1), float(etype))
    ext[:,:-1] = e
    return ext

def get_extrema(x):
    if len(x) == 0: 
        print('Error in get_extrema: Length of x should not be zero.')
        return np.array([])
    x = np.array(x)
    ### Shrink the part of continuous same values
    prune_cond = np.r_[True, x[1:] != x[:-1]]
    w = np.argwhere(prune_cond).reshape(-1)
    e = np.extract(prune_cond, x)
    if len(e) == 1:
        ### x contains only one value. Set the value to be a maximum.
        return np.array([[0.0, e[0], 1.0]])
    x_prune = np.array([w,e]).T
    ### Extract max and min
    max_cond = np.r_[True, x_prune[1:,-1] > x_prune[:-1,-1]] & \
               np.r_[x_prune[1:,-1] < x_prune[:-1,-1], True]
    x_max = _generate_extrema(x_prune, max_cond, 1.0)
    min_cond = np.r_[True, x_prune[1:,-1] < x_prune[:-1,-1]] & \
               np.r_[x_prune[1:,-1] > x_prune[:-1,-1], True]
    x_min = _generate_extrema(x_prune, min_cond, -1.0)
    ### Concatenate max min and sort with original args
    x_ext = np.concatenate((x_max, x_min), axis=0)
    x_ext = x_ext[x_ext[:,0].argsort()]
    return x_ext
