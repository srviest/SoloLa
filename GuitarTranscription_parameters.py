#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/03/15
--------------------------------------------------------------------------------
Script for transcribing audio into sheet music.
The pipeline are as follows:
--------------------------------------------------------------------------------
    S0. Monaural source separation
    S1. Melody extraction
    S2. Note tracking
    S3. Expression style recognition
    S4. Fingering arrangement
--------------------------------------------------------------------------------
"""

# melody contour
contour_hop = 256
contour_sr = 44100
mean_filter_size = 5
guessUnvoiced = True
voiceVibrato = True
voicingTolerance = 0.2
binResolution = 10
minDuration = 100
harmonicWeight = 0.8

# long slide detection
max_transition_note_duration = 0.09
min_transition_note_duration = 0.015


# feature extraction
selected_features = ['hfc', 'pitch', 'pitch_instantaneous_confidence', 
            'pitch_salience', 'silence_rate_20dB', 'silence_rate_30dB', 
            'silence_rate_60dB', 'spectral_complexity', 'spectral_crest', 
            'spectral_decrease', 'spectral_energy', 'spectral_energyband_low', 
            'spectral_energyband_middle_low', 'spectral_energyband_middle_high', 
            'spectral_energyband_high', 'spectral_flatness_db', 'spectral_flux', 
            'spectral_rms', 'spectral_rolloff', 'spectral_strongpeak', 
            'zerocrossingrate', 'inharmonicity', 'tristimulus', \
            'oddtoevenharmonicenergyratio']

# data preprocessing
data_preprocessing_method = []
