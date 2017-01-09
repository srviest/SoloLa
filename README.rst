.. image:: http://yp-chen.com/images/SoloLa_logo.png
================================================================================
SoloLa!
================================================================================

SoloLa! is an automatic system for transforiming lead guitar audio signal in music 
recording into sheet music, which features automatic guitar expression style recognition.

SoloLa! applies a number of music signal processing and machine learning algorithms to achieve our goal — transcribing lead guitar in mixed recording into sheet music with just a click of the mouse. The figure shows the workflow of SoloLa!. The audio signal of lead guitar is isolated from the music mixture by the process of Monaural source separation. For the isolated lead guitar audio signal, Melody extraction aims at automatically estimating the fundamental frequency corresponding to the pitch of the lead guitar to generate a series of consecutive pitch values which are continuous in both time and frequency, a.k.a. melody contour. Expression style recognition refers to the detection of applied lead guitar playing techniques such as string bend, slide or vibrato. Note tracking is the task of recognizing the note event from the estimated frame-level melody contour estimated in the melody extraction stage. The purpose of the task is to transform the mid-level melody contour into high-level symbolic notation. Finally, Automatic fingering arrangement maps the sequence of notes to a set of guitar fretboard positions.

The system comprises of the following processing bloakcs:
	0. Downbeat tracking - estimate downbeat time instants
	1. Monaural source separation - isolate the audio signal of guitar solo from mixture
	2. Melody extraction - estimate the fundamental frequency corresponding to the pitch of the lead guitar to generate a series of consecutive pitch values which are continuous in both time and frequency, a.k.a. melody contour
	3. Note tracking - track the estimated melody contour to recognize discrete musical note events 
	4. Expression style recognition - the detection of applied lead guitar playing techniques such as string bend, slide and vibrato
	5. Fingering arrangement - maps the sequence of notes to a set of guitar fretboard positions

.. image:: https://github.com/srviest/SoloLa-/blob/master/System_overview.jpeg

Requirements
------------
- `numpy <http://www.numpy.org>`_
- `mir_eval <https://github.com/craffel/mir_eval>`_
- `cython <http://www.cython.org>`_
- `scipy <http://www.scipy.org>`_
- `nose <https://github.com/nose-devs/nose>`_
- `madmom <https://github.com/CPJKU/madmom>`_
- `scikit-learn (0.17.0) <http://scikit-learn.org/stable/>`_
- `networkx <https://networkx.github.io/>`_
- `ESSENTIA (2.1-dev) <http://essentia.upf.edu/>`_



Author
------

Yuan-Ping Chen


Basic Usage
------

Example: 
python GuitarTranscrption_script.py ./Input_audio.wav ./Result

(the detail is in python GuitarTranscription_script.py -h.)


References
----------

.. [1] Zafar Rafii and Bryan Pardo,
    *REpeating Pattern Extraction Technique (REPET): A Simple Method for Music/Voice Separation*,
    IEEE Transactions on Audio, Speech, and Language Processing, 21(1):71--82, January 2013.
 
.. [2] Derry FitzGerald, 
    *Harmonic/Percussive Separation using Median Filtering*,
    in Proc. of the International Conference on Digital Audio Effects (DAFx), 2010.
 
.. [3] J. Salamon and E. Gómez. 
    *Melody Extraction from Polyphonic Music Signals using Pitch Contour Characteristics*,
    IEEE Transactions on Audio, Speech and Language Processing, 20(6):1759-1770, Aug. 2012.

.. [4] Gregory Burlet and Ichiro Fujinaga,
    *Robotaba Guitar Tablature Transcription Framework*, 
    in Proc. of the 14th International Society for Music Information Retrieval Conference (ISMIR), 2013.
 
.. [5] M. Mauch and S. Dixon. 
    *pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions*, 
    in Proc. of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2014.
 
.. [6] L. Su, L.-F. Yu and Y.-H. Yang.
    *Sparse Cepstral and Phase Codes for Guitar Playing Technique Classification*, 
    in Proc. of the 15th International Society for Music Information Retrieval Conference (ISMIR), 2014.
 
.. [7] Y.-P. Chen, L. Su and Y.-H. Yang.
    *Electric Guitar Playing Technique Detection in Real-World Recording Based on F0 Sequence Pattern Recognition*, 
    in Proc. of the 16th International Society for Music Information Retrieval Conference (ISMIR), 2015.
 
.. [8] J. Driedger and M. Müller.
    *TSM Toolbox: MATLAB Implementations of Time-Scale Modification Algorithms*, 
    in Proc. of the International Conference on Digital Audio Effects (DAFx), 2014.
 
.. [9] B. McFee, E. Humphrey, and J.P. Bello,
    *A software framework for musical data augmentation*, 
    in Proc. of the 16th International Society for Music Information Retrieval Conference (ISMIR), 2015.

.. [10] Florian Krebs, Sebastian Böck and Gerhard Widmer, 
	*An Efficient State Space Model for Joint Tempo and Meter Tracking*, 
	in Proc. of the 16th International Society for Music Information Retrieval Conference (ISMIR), 2015.

.. [11] Florian Krebs, Sebastian Böck and Gerhard Widmer, 
    *Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical Audio*,
    in Proc. of the 14th International Society for Music Information Retrieval Conference (ISMIR), 2013.
