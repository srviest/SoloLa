#!/bin/bash

# The script install the following libraries:
# -------------------------------------------
# Essentia
# cython
# nose
# Madmom
# scikit-learn
# networkx

path=$(pwd)
cd ${path}
pip install cython
pip install nose
pip install mir_eval
pip install madmom
pip install -U scikit-learn
pip install networkx
brew tap MTG/essentia
brew install essentia --HEAD