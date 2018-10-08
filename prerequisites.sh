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
pip3 install --upgrade pip
pip3 install cython
pip3 install nose
pip3 install mir_eval
pip3 install -U scikit-learn
pip3 install networkx
pip3 install librosa
pip3 install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip3 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip3 install builtins
#pip3 install theano==1.0.3
#pip3 install lasagne==0.2.dev1
#brew tap MTG/essentia
#brew install essentia --HEAD
