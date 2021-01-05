"""
Author: Yuan-Ping Chen, Ting-Wei Su
Date: 2016/04/24
--------------------------------------------------------------------------------
Script for training guitar playing technique classification models
--------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import unicode_literals
from builtins import zip
from builtins import str
from builtins import range
import glob, os, sys, fnmatch, time, random, csv
import numpy as np
import librosa as rosa
import theano
import theano.tensor as T
import lasagne
import pprint
from guitar_trans import models
from guitar_trans import parameters as pm
from lasagne import layers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

model_dir = "model"
output_dir = "outputs"

#=====LOAD AND PREPROCESS INPUT FEATURES=====#

def replace_leading_ending_zeros(array):
    for idx, a in enumerate(array):
        if a > 0:
            array[:idx] = array[idx]
            break
    for idx, a in enumerate(reversed(array)):
        if a > 0:
            i = len(array)-1-idx
            array[i:] = array[i]
            break

def save_to_feature_bank(bank, feature, num):
    num = int(num)
    for idx, cv in enumerate(pm.cv_list):
        if num in cv:
            bank[idx].append(feature)
            return

def load_n_preprocess_input_feature(audio_dir, mc_dir, m_class, sep_direction=False):
    assert os.path.isdir(audio_dir), \
           "{} is not a directory.".format(audio_dir)
    assert os.path.isdir(mc_dir), \
           "{} is not a directory.".format(mc_dir)

    print('Loading data...')
    start_time = time.time()
    if sep_direction:
        feature_bank = {pm.D_ASCENDING: [], pm.D_DESCENDING: []}
        for k in feature_bank:
            feature_bank[k] = [[] for _ in pm.cv_list]
        cls_len = { pm.D_ASCENDING: np.zeros(pm.NUM_CLASS, dtype=int),
                    pm.D_DESCENDING: np.zeros(pm.NUM_CLASS, dtype=int)}
    else:
        feature_bank = { pm.D_ASCENDING: [] }
        for k in feature_bank:
            feature_bank[k] = [[] for _ in pm.cv_list]
        cls_len = { pm.D_ASCENDING: np.zeros(pm.NUM_CLASS, dtype=int) }
    for root, dirs, files in os.walk(audio_dir):
        for fi in files:
            ### Load features
            if '.wav' in fi:
                # print('file name: {}'.format(fi))
                y, sr = rosa.load(os.path.join(root, fi), sr=pm.SAMPLING_RATE, mono=True)
                fn = os.path.splitext(fi)[0]
                mc = np.loadtxt(mc_dir+'/'+fn+'.MIDI.melody', dtype='float32')
                
                ### Preprocess melody contour
                if len(mc) < 18:
                    print('{} mc length must be larger than 18. (only {}).'.format(fi, len(mc)))
                    continue
                elif len(mc) < pm.MC_LENGTH:
                    mc = np.pad(mc, (0, pm.MC_LENGTH-len(mc)), 'edge')
                elif len(mc) > pm.MC_LENGTH:
                    mc = mc[:pm.MC_LENGTH]
                replace_leading_ending_zeros(mc)
                
                ### Classify ascending or descending
                if sep_direction:
                    if fn.split('_')[0] == pm.HAMM:
                        direction = pm.D_ASCENDING
                    elif fn.split('_')[0] == pm.PULL:
                        direction = pm.D_DESCENDING
                    elif mc[:5].mean() <= mc[-5:].mean():
                        direction = pm.D_ASCENDING
                    else:
                        direction = pm.D_DESCENDING
                    c_class = fn.split('_')[0]
                else:
                    direction = pm.D_ASCENDING
                    c_class = pm.HAMM if fn.split('_')[0] == pm.PULL else fn.split('_')[0]
                bank = feature_bank[direction]

                ### Create the answer in a form like [0,0,0,1,0]
                ans_num = pm.tech_dict[direction][c_class]
                cls_len[direction][int(ans_num)] += 1
                ans = np.zeros(pm.NUM_CLASS, dtype='int32')
                ans[ans_num] = 1

                ### Extract feature
                feature = m_class.extract_features(y, mc, fn, ans)
                if feature is None: continue
                save_to_feature_bank(bank, feature, int(fn.split('_')[2]))
    print('Totally loaded {} secs.'.format(time.time()-start_time))
    print('Class lengths: {}'.format(cls_len))
    return feature_bank

#=====DATA DISTRIBUTION=====#

def balance_number_of_data(data_list):
    clss = [[] for i in range(pm.NUM_CLASS)]
    for dt in data_list:
        clss[np.argmax(dt[-2])].append(dt)
    min_len = min([len(c) for c in clss])
    print('Balance each class to {} data.'.format(min_len))
    new_data_list = []
    for c in clss:
        new_data_list += random.sample(c, min_len)
    random.shuffle(new_data_list)
    return new_data_list

def get_train_test_feat(feature_bank, idx, balance=False):
    train_list, test_list = [], []
    for i in range(len(feature_bank)):
        if i == idx:
            test_list += feature_bank[i]
        else:
            train_list += feature_bank[i]
    if balance: 
      train_list = balance_number_of_data(train_list)
    np.random.shuffle(train_list)
    return train_list, test_list

#=====CLASSIFICATION=====#

def classify(feature_bank, model_name, model_class, param_set, sep_direction=True, test_aug=False):
    if not os.path.isdir(os.path.join(model_dir, model_name)):
        os.mkdir(os.path.join(model_dir, model_name))
    if not os.path.isdir(os.path.join(output_dir, model_name)):
        os.mkdir(os.path.join(output_dir, model_name))
    all_results = {}
    for key in feature_bank:
        direction_type = key if sep_direction else pm.D_MIXED
        print('Training {}s...'.format(direction_type))
        bank = feature_bank[key]
        cm_all = np.zeros((pm.NUM_CLASS, pm.NUM_CLASS), dtype=int)
        for idx in range(len(bank)):
            model_file = model_name+'_'+str(idx)+'.'+direction_type+'.npz'
            model_fp = os.path.join(model_dir, model_name, model_file)
            train_list, test_list = get_train_test_feat(bank, idx, balance=False)

            ### initialize model
            model = model_class(param_set, model_fp)

            ### train model and save training result
            model.train(train_list, 100)

            ### test and evaluate
            npzfile = np.load(model_fp, allow_pickle=True)
            model.set_param_values(npzfile['params'])
            if test_aug:
                cm = model.test(test_list)
            else:
                origin_test_list = []
                for t in test_list:
                    if 'aug' not in t[-1]:
                        origin_test_list.append(t)
                cm = model.test(origin_test_list)
            cm_all += cm
        
        
        csv_fn = 'evaluation.' + direction_type + '.csv'
        save_fp = os.path.join(output_dir, model_name, csv_fn)
        eval_scores(cm_all, key, print_scores=True, save_fp=save_fp)
        all_results[key] = cm_all
    return all_results

#=====EVALUATION=====#

def eval_scores(cm, direction_type, print_scores=True, save_fp=None):
    t, p = [], []
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            for _ in range(cm[i][j]):
                t.append(i)
                p.append(j)
    each_p = precision_score(t, p, average=None)
    each_r = recall_score(t, p, average=None)
    each_f = f1_score(t, p, average=None)
    all_p = precision_score(t, p, average='weighted')
    all_r = recall_score(t, p, average='weighted')
    all_f = f1_score(t, p, average='weighted')

    final_acc = float(np.sum(np.diagonal(cm))) * 100 / float(np.sum(cm))

    dt = pm.inv_tech_dict[direction_type]
    score_list = ["Precision", "Recall", "F1"]
    row_format_1 = "{:>8}" + "{:>12}" * len(score_list)
    row_format_2 = "{:>8}" + "{:>12.4f}" * len(score_list)

    if print_scores: 
        print('Accuracy: {:.2f} %'.format(final_acc))
        print('Confusion Matrix:')
        print(cm)
        print('')
        print('Scores:')
        print(row_format_1.format("", *score_list))
    scores = [[""] + score_list]
    for idx, _p, _r, _f in zip(list(range(len(each_p))), each_p, each_r, each_f):
        if print_scores: print(row_format_2.format(dt[idx], _p, _r, _f))
        scores.append([dt[idx], "{:.4f}".format(_p), "{:.4f}".format(_r), "{:.4f}".format(_f)])
    if print_scores: print(row_format_2.format("All", all_p, all_r, all_f))
    scores.append(["All", "{:.4f}".format(all_p), "{:.4f}".format(all_r), "{:.4f}".format(all_f)])
    if save_fp is not None:
        ### Save as a csv file
        cm_table = np.hstack(([[dt[i]] for i in range(pm.NUM_CLASS)], cm))
        cm_table = np.vstack(([[''] + [dt[i] for i in range(pm.NUM_CLASS)]], cm_table))
        data = cm_table.tolist() + [['Accuracy', '{:.2f} %'.format(final_acc)], ['---']] + scores
        
        csv_fi = open(save_fp, 'w')
        w = csv.writer(csv_fi, delimiter = ',')
        for r in data:
            w.writerow(r)
        csv_fi.close()
    return scores

#=====MAIN FUNCTION=====#

def main(model_name, model_type, model_opts, data_dir, sep_direction=True, test_aug=False, description=None):
    if description is not None:
        print('Description: {}'.format(description))
    audio_dir = os.path.join(data_dir, 'audio')
    mc_dir = os.path.join(data_dir, 'melody')
    model_class = getattr(models, model_type)
    param_set = getattr(pm, model_opts)
    ### load and pre-process input features
    # feature_bank = load_n_preprocess_input_feature(audio_dir, mc_dir, model_class, sep_direction)
    # np.save('feature_bank_mfcc.npy', feature_bank)
    feature_bank = np.load('feature_bank_mfcc.npy', allow_pickle=True).item()
    all_results = classify(feature_bank, model_name, model_class, param_set, sep_direction=True, test_aug=False)
    return all_results


def parser():
    import argparse
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description=
    """
===================================================================
Script for training guitar playing technique classification models.
===================================================================
    """)

    p.add_argument('model_name', type=str, metavar='model_name',
                    help='The name of this new model.')
    p.add_argument('model_type', type=str, metavar='model_type',
                    help='The type of this new model. The types are the classes defined in models.py. See models.py for more information.')
    p.add_argument('model_opts', type=str, metavar='model_opts',
                    help='The name of parameter dictionary of this new model. This parameter dictionary should be defined in parameters.py.')
    p.add_argument('data_dir', type=str, metavar='data_dir',
                    help='The directory of the dataset to be used.')
    p.add_argument('-d', '--description', type=str, 
                    help='The description of this model.')
    return p.parse_args()

if __name__ == '__main__':
    args = parser()
    main(args.model_name, args.model_type, args.model_opts, args.data_dir, description=args.description)

