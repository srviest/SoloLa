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
pipenv install cython
pipenv install nose
pipenv install mir_eval
pipenv install -U scikit-learn
pipenv install networkx
pipenv install librosa
pipenv install theano
pipenv install lasagne
brew tap MTG/essentia
brew install essentia --HEAD
