import librosa as rosa
import numpy as np
import guitar_trans.te_note_tracking as note_tracking
import guitar_trans.parameters as pm
from guitar_trans import models
from guitar_trans.song import *
from guitar_trans.note import *
from guitar_trans.contour import *
from guitar_trans.technique import *
from guitar_trans.evaluation import evaluation_note, evaluation_esn, evaluation_ts
from melody_extraction import extract_melody
from os import path, sep, makedirs

N_BIN = int(round(0.14 * 44100))
N_FRAME = pm.MC_LENGTH

def transcribe(audio, melody, asc_model_fp, desc_model_fp, save_dir, audio_fn):
    if not path.exists(save_dir): makedirs(save_dir)
    print '  Output directory: ', '\n', '    ', save_dir
    trend, new_melody, notes = note_tracking.tent(melody, debug=save_dir)
    np.savetxt(save_dir+sep+'FilteredMelody.txt', new_melody.seq, fmt='%.8f')
    np.savetxt(save_dir+sep+'TentNotes.txt', [n.discrete_to_cont(pm.HOP_LENGTH, pm.SAMPLING_RATE).array_repr() for n in notes], fmt='%.8f')
    cand_dict = {pm.D_ASCENDING: [], pm.D_DESCENDING: []}
    cand_ranges = []
    rate = float(pm.HOP_LENGTH) / float(pm.SAMPLING_RATE)
    for nt in notes:
        if isinstance(nt, CandidateNote):
            for seg in nt.segs:
                mid_frame = nt.onset + seg.mid
                mid_bin = int(float(mid_frame) / rate)
                start_i, end_i = mid_frame - N_FRAME/2, mid_frame + N_FRAME - N_FRAME/2
                start_bin = start_i * pm.HOP_LENGTH
                sub_audio = audio[start_bin: start_bin + N_BIN]
                sub_mc = melody[start_i: end_i]
                assert(len(sub_audio) == N_BIN)
                assert(len(sub_mc) == N_FRAME)
                sub_fn = audio_fn + '_' + str(mid_frame)
                direction = pm.D_ASCENDING if seg.val >= 0 else pm.D_DESCENDING
                cand_dict[direction].append((sub_audio, sub_mc, sub_fn, nt, seg, start_i, end_i))
                # rosa.output.write_wav('trans/audio/clip_'+sub_fn+'.wav', sub_audio, sr=pm.SAMPLING_RATE, norm=False)
    cand_results = []
    no_next = []
    for direction in cand_dict:
        cand_list = cand_dict[direction]
        model_fp = asc_model_fp if direction == pm.D_ASCENDING else desc_model_fp
        if len(cand_list) > 0:
            pred_list = classification(model_fp, [cand[:3] for cand in cand_list])
            for pred, cand in zip(pred_list, cand_list):
                sub_audio, sub_mc, sub_fn, nt, seg, start_i, end_i = cand
                t_name = pm.inv_tech_dict[direction][np.argmax(pred)]
                t_type = get_tech(t_name, direction)
                t_val = int(round(seg.diff())) if t_type in (T_BEND, T_RELEASE) else 1
                if t_type < T_NORMAL:
                    ### Merge Notes
                    if nt.next_note is None:
                        print 'No next note. Ignore this candidate.'
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

def main(audio_fp, asc_model_fp, desc_model_fp, output_dir, mc_fp=None, eval_note=None, eval_ts=None):
    audio_fn = path.splitext(path.basename(audio_fp))[0]
    save_dir = path.join(output_dir, audio_fn)
    if mc_fp is None:
        mc, mc_midi = extract_melody(audio_fp, save_dir)
    else:
        mc_midi = np.loadtxt(mc_fp)
    audio, sr = rosa.load(audio_fp, sr=None, mono=True)
    melody = Contour(0, mc_midi)
    notes = transcribe(audio, melody, asc_model_fp, desc_model_fp, save_dir, audio_fn)
    if eval_note is not None:
        sg = Song(name=audio_fn)
        sg.load_esn_list(eval_note)
        evaluation_note(sg.es_note_list, notes, save_dir, audio_fn, string='evaluate notes')
        evaluation_esn(sg.es_note_list, notes, save_dir, audio_fn, string='evaluate esn')

    if eval_ts is not None:
        ans_list = np.loadtxt(eval_ts)
        # TODO

def parser():
    import argparse
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description=
    """
===================================================================
Script for transcribing a song.
===================================================================
    """)
    p.add_argument('audio_fp', type=str, metavar='audio_fp',
                    help='The filepath of the audio to be transcribed.')
    p.add_argument('asc_model_fp', type=str, metavar='asc_model_fp',
                    help='The name of the ascending model.')
    p.add_argument('desc_model_fp', type=str, metavar='desc_model_fp',
                    help='The name of the descending model.')
    p.add_argument('output_dir', type=str, metavar='output_dir', default='outputs',
                    help='The output directory.')
    p.add_argument('-m', '--melody_contour', type=str, default=None, 
                    help='The filepath of melody contour.')
    p.add_argument('-e', '--evaluate', type=str, default=None, 
                    help='The filepath of answer file.')


if __name__ == '__main__':
    args = parser()
    main(args.audio_fp, args.asc_model_fp, args.desc_model_fp, 
         args.output_dir, args.melody_contour, args.evaluate)

