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
brew install pkg-config gcc readline sqlite gdbm freetype libpng
brew install libyaml fftw ffmpeg libsamplerate libtag
brew install python --framework
pip install ipython numpy matplotlib pyyaml
./waf configure --mode=release --build-static --with-python --with-cpptests --with-examples --with-vamp --with-gaia
./waf
./waf install