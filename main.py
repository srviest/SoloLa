import te_note_tracking as note_tracking
import models
import parameters as pm
import librosa as rosa
import note
import numpy as np
from contour import *
from technique import *
from os import path

N_BIN = int(round(0.14 * 44100))
N_FRAME = pm.MC_LENGTH

def transcribe(audio_fp, melody_fp, asc_model_fp, desc_model_fp):
    fn = path.splitext(path.basename(audio_fp))[0]
    y, sr = rosa.load(audio_fp, sr=None, mono=True)
    mel = np.loadtxt(melody_fp)
    melody = Contour(0, mel)
    trend, new_melody, notes = note_tracking.tent(melody)
    np.savetxt('trans/new_melody.txt', new_melody.seq, fmt='%.8f')
    np.savetxt('trans/nt1.txt', [n.discrete_to_cont(pm.HOP_LENGTH, pm.SAMPLING_RATE).array_repr() for n in notes], fmt='%.8f')
    cand_dict = {pm.D_ASCENDING: [], pm.D_DESCENDING: []}
    cand_ranges = []
    rate = float(pm.HOP_LENGTH) / float(pm.SAMPLING_RATE)
    for nt in notes:
        if isinstance(nt, note.CandidateNote):
            for seg in nt.segs:
                mid_frame = nt.onset + seg.mid
                mid_bin = int(float(mid_frame) / rate)
                start_i, end_i = mid_frame - N_FRAME/2, mid_frame + N_FRAME - N_FRAME/2
                start_bin = start_i * pm.HOP_LENGTH
                sub_audio = y[start_bin: start_bin + N_BIN]
                sub_mc = melody[start_i: end_i]
                assert(len(sub_audio) == N_BIN)
                assert(len(sub_mc) == N_FRAME)
                sub_fn = fn + '_' + str(mid_frame)
                direction = pm.D_ASCENDING if seg.val >= 0 else pm.D_DESCENDING
                cand_dict[direction].append((sub_audio, sub_mc, sub_fn, nt, seg, start_i, end_i))
                # rosa.output.write_wav('trans/audio/clip_'+sub_fn+'.wav', sub_audio, sr=pm.SAMPLING_RATE, norm=False)
    # np.savetxt('cand_ranges.txt', cand_ranges)
    # np.save('cand_dict.npy', cand_dict)
    # cand_dict = np.load('cand_dict.npy').item()
    cand_results = []
    for direction in cand_dict:
        cand_list = cand_dict[direction]
        model_fp = asc_model_fp if direction == pm.D_ASCENDING else desc_model_fp
        if len(cand_list) > 0:
            pred_list = classification(model_fp, [cand[:3] for cand in cand_list])
            for pred, cand in zip(pred_list, cand_list):
                t_name = pm.inv_tech_dict[direction][np.argmax(pred)]
                t_type = get_tech(t_name, direction)
                t_val = int(round(cand[4].diff())) if t_type in (T_BEND, T_RELEASE) else 1
                nt = cand[3]
                if t_type < T_NORMAL:
                    nt.add_tech(Tech(t_type, t_val))
                sign = 1 if direction == pm.D_ASCENDING else -1 
                cand_results.append([cand[5] * rate, cand[6] * rate, t_type * sign])
    np.savetxt('trans/cand_results.txt', cand_results, fmt='%.8f')
    note.merge_notes(notes)
    cont_notes = [nt.discrete_to_cont(pm.HOP_LENGTH, pm.SAMPLING_RATE) for nt in notes]
    np.savetxt('trans/nt2.txt', [n.array_repr() for n in cont_notes], fmt='%.8f')
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

