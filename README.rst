.. image:: http://yp-chen.com/images/SoloLa_logo.png
================================================================================
================================================================================


SoloLa! is an automatic system for transforming lead guitar audio signal in music 
recording into sheet music, which features automatic guitar expression style recognition.

The system comprises of the following processing bloakcs:
	1. Source Separation - isolate the audio signal of guitar solo from mixture
	2. Melody Extraction - estimate the fundamental frequency corresponding to the pitch of the lead guitar to generate a series of consecutive pitch values which are continuous in both time and frequency, a.k.a. melody contour
	3. Note Tracking - track the estimated melody contour to recognize discrete musical note events 
	4. Expression Style Recognition - the detection of applied lead guitar playing techniques such as string bend, slide and vibrato
	5. Fingering Arrangement - maps the sequence of notes to a set of guitar fretboard positions

.. image:: https://raw.githubusercontent.com/SoloLa-Platform/SoloLa/master/solola_workflow.jpg

- Melody contour plot reproduced from J. Salamon, E. Gómez, D. P. W. Ellis and G. Richard, "Melody Extraction from Polyphonic Music Signals: Approaches, Applications and Challenges", IEEE Signal Processing Magazine, 31(2):118-134, Mar. 2014 with permission from the authors.

Requirements
------------
- `numpy <http://www.numpy.org>`_
- `scipy <http://www.scipy.org>`_
- `ESSENTIA <http://essentia.upf.edu/>`_
- `librosa <http://librosa.github.io/librosa/index.html>`_
- `theano <http://deeplearning.net/software/theano/>`_
- `Lasagne <http://lasagne.readthedocs.io/en/latest/>`_
- `scikit-learn <http://scikit-learn.org/stable/>`_
- `mir_eval <https://github.com/craffel/mir_eval>`_
.. - `cython <http://www.cython.org>`_
.. - `nose <https://github.com/nose-devs/nose>`_
.. - `networkx <https://networkx.github.io/>`_
.. - `madmom <https://github.com/CPJKU/madmom>`_



Author
------

Yuan-Ping Chen, Ting-Wei Su


.. Basic Usage
.. ------

.. ``$ python GuitarTranscrption_script.py ./Input_audio.wav ./Result``

.. (the detail is in python GuitarTranscription_script.py -h.)


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
