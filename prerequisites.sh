#!/bin/bash

# The script install the following libraries:
# -------------------------------------------
# Essentia
# cython
# nose
# scikit-learn
# networkx
# librosa
# theano
# lasagne

path=$(pwd)
cd ${path}
pip install cython
pip install nose
pip install mir_eval
pip install -U scikit-learn
pip install networkx
pip install librosa
pip install theano
pip install lasagne
brew tap MTG/essentia
brew install essentia --HEAD
