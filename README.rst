================================================================================
SoloLa!
================================================================================

SoloLa! is an automatic system for transforiming lead guitar audio signal in music 
recording into sheet music, which features the guitar expression style recognition.

The system comprises of the following algorithms:
	0. Downbeat tracking
	1. Monaural source separation
	2. Melody extraction
	3. Note tracking 
	4. Expression style recognition
	5. Fingering arrangement


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

License
-------

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




Acknowledgements
----------------


