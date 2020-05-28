
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from builtins import zip
from builtins import str
from past.utils import old_div
import librosa as rosa
import numpy as np
import guitar_trans.te_note_tracking as note_tracking
import guitar_trans.parameters as pm
from guitar_trans import models
from guitar_trans.song import *
from guitar_trans.note import *
# Contour()
from guitar_trans.contour import *
from guitar_trans.technique import *
from guitar_trans.evaluation import evaluation_note, evaluation_esn, evaluation_ts
from os import path, sep, makedirs

# melody components
from melody.melody_extraction import extract_melody

N_BIN = int(round(0.14 * 44100))
N_FRAME = pm.MC_LENGTH
def transcribe(audio, melody, asc_model_fp, desc_model_fp, save_dir, audio_fn):
    if not path.exists(save_dir): makedirs(save_dir)
    print('  Output directory: ', '\n', '    ', save_dir)
    trend, new_melody, notes = note_tracking.tent(melody, debug=save_dir)
    np.savetxt(save_dir+sep+'FilteredMelody.txt', new_melody.seq, fmt='%.8f')
    np.savetxt(save_dir+sep+'TentNotes.txt', [n.discrete_to_cont(pm.HOP_LENGTH, pm.SAMPLING_RATE).array_repr() for n in notes], fmt='%.8f')
    
    cand_dict = {pm.D_ASCENDING: [], pm.D_DESCENDING: []}
    cand_ranges = []
    rate = old_div(float(pm.HOP_LENGTH), float(pm.SAMPLING_RATE))
    cand_results = []
    for nt in notes:
        if nt.tech(T_BEND).value > 0:
            cand_results.append([nt.onset * rate, nt.offset * rate, T_BEND])
        if nt.tech(T_RELEASE).value > 0:
            cand_results.append([nt.onset * rate, nt.offset * rate, -T_RELEASE])
        if nt.tech(T_SLIDE_IN).value > 0:
            cand_results.append([nt.onset * rate, nt.offset * rate, T_SLIDE_IN])
        if nt.tech(T_SLIDE_OUT).value > 0:
            cand_results.append([nt.onset * rate, nt.offset * rate, T_SLIDE_OUT])
        if nt.tech(T_VIBRATO).value > 0:
            cand_results.append([nt.onset * rate, nt.offset * rate, T_VIBRATO])
        for seg in nt.segs:
            mid_frame = nt.onset + seg.mid
            mid_bin = int(old_div(float(mid_frame), rate))
            start_i, end_i = mid_frame - old_div(N_FRAME,2), mid_frame + N_FRAME - old_div(N_FRAME,2)
            start_bin = start_i * pm.HOP_LENGTH
            sub_audio = audio[start_bin: start_bin + N_BIN]
            sub_mc = melody[start_i: end_i]
            assert(len(sub_audio) == N_BIN)
            assert(len(sub_mc) == N_FRAME)
            sub_fn = audio_fn + '_' + str(mid_frame)
            direction = pm.D_ASCENDING if seg.val >= 0 else pm.D_DESCENDING
            cand_dict[direction].append((sub_audio, sub_mc, sub_fn, nt, seg, start_i, end_i))
            # rosa.output.write_wav('trans/audio/clip_'+sub_fn+'.wav', sub_audio, sr=pm.SAMPLING_RATE, norm=False)
    no_next = []
    for direction in cand_dict:
        print('Processing direction', direction)
        cand_list = cand_dict[direction]
        model_fp = asc_model_fp if direction == pm.D_ASCENDING else desc_model_fp
        if len(cand_list) > 0:
            pred_list = classification(model_fp, [cand[:3] for cand in cand_list])
            for pred, cand in zip(pred_list, cand_list):
                sub_audio, sub_mc, sub_fn, nt, seg, start_i, end_i = cand
                t_name = pm.inv_tech_dict[direction][np.argmax(pred)]
                t_type = get_tech(t_name, direction)
                origin_t_val = nt.tech(t_type).value
                t_val = int(round(seg.diff())) if t_type in (T_BEND, T_RELEASE) else origin_t_val + 1
                if t_type < T_NORMAL:
                    ### Merge Notes
                    if nt.next_note is None:
                        print('No next note. Ignore this candidate.')
                        no_next.append([start_i * rate, end_i * rate, t_type * sign])
                        continue
                        # print 'next_note is None'
                        # print nt, cand[4]
                        # if t_type in [T_HAMMER, T_PULL, T_SLIDE]:
                        #     print('WARNING!!! Changing {} to bend or release.'.format(t_type))
                        #     print cand[4]
                        #     t_type = T_BEND if direction == pm.D_ASCENDING else T_RELEASE
                    elif t_type in [T_BEND, T_RELEASE]:
                        if nt.next_note in notes:
                            notes.remove(nt.next_note)
                        nt.merge_note(nt.next_note)
                    elif t_type in [T_HAMMER, T_PULL, T_SLIDE]:
                        tv = nt.next_note.tech(t_type).value
                        nt.next_note.add_tech(Tech(t_type, tv+2))
                    nt.add_tech(Tech(t_type, t_val))
                sign = 1 if direction == pm.D_ASCENDING else -1 
                cand_results.append([start_i * rate, end_i * rate, t_type * sign])
    np.savetxt(save_dir+sep+'NoNextNote.txt', no_next, fmt='%.8f')
    np.savetxt(save_dir+sep+'CandidateResults.txt', cand_results, fmt='%.8f')
    # note.merge_notes(notes)
    cont_notes = [nt.discrete_to_cont(pm.HOP_LENGTH, pm.SAMPLING_RATE) for nt in notes]
    np.savetxt(save_dir+sep+'FinalNotes.txt', [n.array_repr() for n in cont_notes], fmt='%.8f')
    return cont_notes
            
def classification(model_fp, cand_list):
    model = models.Model.init_from_file(model_fp)
    data_list = [model.extract_features(*(cand[:3])) for cand in cand_list]
    pred_list = model.run(data_list)
    return pred_list   

def get_tech(t_name, direction):
    if t_name == pm.BEND and direction == pm.D_ASCENDING:
        return T_BEND
    elif t_name == pm.BEND and direction == pm.D_DESCENDING:
        return T_RELEASE
    elif t_name == pm.HAMM:
        return T_HAMMER
    elif t_name == pm.PULL:
        return T_PULL
    elif t_name == pm.SLIDE:
        return T_SLIDE
    elif t_name == pm.NORMAL:
        return T_NORMAL
    else:
        raise ValueError("t_name shouldn't be {}.".format(t_name))