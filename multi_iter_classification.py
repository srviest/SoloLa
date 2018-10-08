from __future__ import print_function
from __future__ import unicode_literals
from builtins import str
from builtins import range
from guitar_trans import models
from guitar_trans import parameters as pm
import classification as clf
import numpy as np
import sys, os

def main(model_name, model_type, model_opts, data_dir, iteration, sep_direction=True, test_aug=False, description=None):
    results = {}
    for key in [pm.D_ASCENDING, pm.D_DESCENDING]:
        results[key] = np.zeros((pm.NUM_CLASS, pm.NUM_CLASS), dtype=int)
    if description is not None:
        print('Description: {}'.format(description))
    audio_dir = os.path.join(data_dir, 'audio')
    mc_dir = os.path.join(data_dir, 'melody')
    model_class = getattr(models, model_type)
    param_set = getattr(pm, model_opts)
    output_dir = clf.output_dir
    clf.model_dir = os.path.join(clf.model_dir, model_name)
    clf.output_dir = os.path.join(clf.output_dir, model_name)
    if not os.path.isdir(clf.model_dir):
        os.mkdir(clf.model_dir)
    if not os.path.isdir(clf.output_dir):
        os.mkdir(clf.output_dir)
    ### load and pre-process input features
    feature_bank = clf.load_n_preprocess_input_feature(audio_dir, mc_dir, model_class, sep_direction)
    # np.save('feature_bank_spec+mc.npy', feature_bank)
    # feature_bank = np.load('feature_bank_mfcc.npy').item()
    print('Run {} iterations.'.format(iteration))
    for i in range(iteration):
        print('iteration: {}'.format(i))
        cm = clf.classify(feature_bank, model_name + '_' + str(i), model_class, param_set, sep_direction=True, test_aug=False)
        for key in cm:
            if key in results:
                results[key] += cm[key]
    for key in results:
        print('Final result of {}'.format(key))
        csv_fn = 'evaluation.' + key + '.csv'
        save_fp = os.path.join(output_dir, model_name, csv_fn)
        clf.eval_scores(results[key], key, print_scores=True, save_fp=save_fp)

def parser():
    import argparse
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description=
    """
======================================================================================
Script for training guitar playing technique classification models for several rounds.
======================================================================================
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
    p.add_argument('-i', '--iteration', type=int, default=10,
                    help='The description of this model.')

    return p.parse_args()

if __name__ == '__main__':
    args = parser()
    main(args.model_name, args.model_type, args.model_opts, args.data_dir, args.iteration, description=args.description)

