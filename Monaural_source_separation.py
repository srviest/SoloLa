#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/03/11
----------------------------------------------------------------------
Source separation: attenuate the accompaniments in order to emphasize
                   guitar solo.
----------------------------------------------------------------------
Args:
    input_files:    Audio files to be processed. 
                    Only the wav files would be considered.
    output_dir:     Directory for storing the results.

Optional args:
    Please refer to --help.
----------------------------------------------------------------------
Returns:
    Guitar solo
    Accompaniments

"""

import scipy.fftpack as fft
import numpy as np
from scipy.io import wavfile, savemat
from sys import float_info

def repet_ada(x,fs):
    # Default adaptive parameters
    par = [24,12,7]
    # Default repeating period range
    per = [0.8,min(8,par[0]/3.)]
    # Analysis window length in seconds (audio stationary around 40 milliseconds)
    alen = 0.040
    # Analysis window length in samples (power of 2 for faster FFT)
    N = 2**nextpow2(alen*fs)
    # Analysis window (even N and 'periodic' Hamming for constant overlap-add)
    win = np.hamming(N)
    #win = np.reshape(win,(win.shape[0],1)) 
    # Analysis step length (N/2 for constant overlap-add)
    stp = N/2.
    # Cutoff frequency in Hz for the dual high-pass filtering (e.g., singing voice rarely below 100 Hz)
    cof = 100.
    # Cutoff frequency in frequency bins for the dual high-pass filtering (DC component = bin 0)
    cof = np.ceil(cof*(N-1)/fs)
    # Number of samples
    t = x.shape[0]
    # Number of channels
    try:
        # multi channel files
        k = x.shape[1]
    except IndexError:
        # catch mono files
        k = 1
    X = np.empty((win.shape[0],np.ceil((N-stp+x.shape[0])/stp),k),'complex128')
    if k>1:
        # Loop over the channels
        for i in range(k):
        	# Short-Time Fourier Transform (STFT) of channel i
            X[:,:,i] = stft(x[:,i],win,stp)
    else:
        i = 0
        X[:,:,i] = stft(x,win,stp)

    # Magnitude spectrogram (with DC component and without mirrored frequencies)
    V = abs(X[0:N/2+1,:,:])
    # Repeating period in time frames (compensate for STFT zero-padding at the beginning)
    per = map(lambda g: g*fs, per)
    per = np.ceil((per+N/stp-1)/stp)
    # per = np.ceil((per*fs+N/stp-1)/stp)
    # Adaptive window length and step length in time frames
    par[0] = round(par[0]*fs/stp)
    par[1] = round(par[1]*fs/stp)
    # Beat spectrogram of the mean power spectrograms
    B = beat_spectrogram(np.mean(V**2,2),par[0],par[1])
    # Repeating periods in time frames
    P = repeating_periods(B,per)
    y = np.zeros((t,k))

    # Loop over the channels
    for i in range(k):
    	# Repeating mask
        Mi = repeating_mask(V[:,:,i],P,par[2])
        # High-pass filtering of the (dual) non-repeating foreground
        s = 1
        e = 1+cof
        Mi[s:e,:] = 1
        # Mirror the frequencies
        Mj = Mi[1:-1,:]
        Mi = np.concatenate((Mi,Mj[::-1]),0)
        # Estimated repeating background
        yi = istft(Mi*X[:,:,i],win,stp)
        # Truncate to the original length of the mixture
        y[:,i] = yi[0:t]
    if  y.shape[1]==1:
        # multi channel files
        y = y.reshape(y.shape[0])
    return y


"""
nextpow2(N) returns the first P such that 2.^P >= abs(N).  It is
often useful for finding the nearest power of two sequence
length for FFT operations.

"""
def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return m_i

"""
Short-Time Fourier Transform (STFT) using fft
X = stft(x,win,stp);

Input(s):
x: signal [t samples, 1]
win: analysis window [N samples, 1]
stp: analysis step

Output(s):
X: Short-Time Fourier Transform [N bins, m frames]

"""
def stft(x,win,stp):
    # Number of samples
    t = x.shape[0]
    # Analysis window length                                                              
    N = win.shape[0]
    # Number of frames with zero-padding
    m = np.ceil((N-stp+t)/stp)
    # Zero-padding for constant overlap-add
    x = np.r_[np.zeros(N-stp),x,np.zeros(m*stp-t)]
    X = np.zeros((N,m),'complex128')
    # Loop over the frames
    for j in range(int(m)):														
    	s = 0+stp*j
    	e = N+stp*j
    	# Windowing and fft
    	X[:,j] = fft.fft(x[s:e]*win)
    return X

"""
Inverse Short-Time Fourier Transform using ifft
x = istft(X,win,stp);

Input(s):
X: Short-Time Fourier Transform [N bins, m frames]
win: analysis window [N samples, 1]
stp: analysis step

Output(s):
x: signal [t samples, 1]
"""

def istft(X,win,stp):
    # Number of frequency bins and time frames
    N,m = X.shape
    # Length with zero-padding                                                          
    l = (m-1)*stp+N
    x = np.zeros(l)
    # Loop over the frames
    for j in range(int(m)):
    	# Un-windowing and ifft (assuming constant overlap-add)
    		s = 0+stp*j
    		e = N+stp*j
    		x[s:e] = x[s:e]+np.real(fft.ifft(X[:,j]))
    # Remove zero-padding at the beginning
    x = x[0:l-(N-stp)]
    # Remove zero-padding at the end
    x = x[N-stp::]
    # Normalize constant overlap-add using win
    x = x/np.sum(win[0:N:stp])
    return x	



"""
Autocorrelation function using fft according to the WienerKhinchin theorem
C = acorr(X);

Input(s):
X: data matrix [n elements, m vectors]

Output(s):
C: autocorrelation matrix [n lags, m vectors]
"""
def acorr(X):
    n,m = X.shape
    # Zero-padding to twice the length for a proper autocorrelation
    X = np.r_[X,np.zeros((n,m))]
    # Power Spectral Density: PSD(X) = fft(X).*conj(fft(X))
    X = abs(fft.fft(X,axis = 0))**2
    # WienerKhinchin theorem: PSD(X) = fft(acorr(X))
    C = abs(fft.ifft(X,axis = 0))
    # Discard the symmetric part (lags n-1 to 1)
    C = C[0:n,:]
    # Unbiased autocorrelation (lags 0 to n-1)
    T = np.arange(n,0,-1)
    T = T.reshape(T.shape[0],1)
    C = C/np.tile(T, [1,m])
    return C


"""
Beat spectrum using the autocorrelation function
b = beat_spectrum(X);

Input(s):
X: spectrogram [n frequency bins, m time frames]

Output(s):
b: beat spectrum [1, m time lags]
"""

def beat_spectrum(X):
    # Correlogram using acorr [m lags, n bins]
    B = acorr(X.T) 
    g = B.dtype
    print('The dtype of out of acorr() is ',g)                                                             
    # Mean along the frequency bins
    b = np.mean(B,1)
    return b


"""
Beat spectrogram using the beat_spectrum
B = beat_spectrogram(X,w,h);

Input(s):
X: spectrogram [n bins, m frames]
w: time window length
h: hop size

Output(s):
B: beat spectrogram [w lags, m frames] (lags from 0 to w-1)
"""
def beat_spectrogram(X,w,h):
    # Number of frequency bins and time frames
    n,m = X.shape
    # Zero-padding to center windows
    X = np.concatenate((np.zeros((n,np.ceil((w-1.)/2))),X,np.zeros((n,np.floor((w-1.)/2)))),1)
    B = np.zeros((w,m))
    # Loop over the time frames (including the last one)
    for j in range(0,m,int(h))+[m-1]:
        # Beat spectrum of the windowed spectrogram centered on frame j
        s = 0+j
        e = w+j
        B[:,j] = beat_spectrum(X[:,s:e]).T
    return B


"""
Repeating periods from the beat spectrogram
P = repeating_periods(B,r);

Input(s):
B: beat_spectrogram [l lags, m frames]
r: repeating period range in time frames [min lag, max lag]

Output(s):
P: repeating periods in time frames [1, m frames]
"""
def repeating_periods(B,r):
    # Discard lags 0
    B = B[1::,:]
    # Beat spectrogram in the repeating period range
    s = r[0]-1
    e = r[1]
    B = B[s:e,:]
    # Maximum values in the repeating period range for all the frames
    P = np.argmax(B,0)
    # The repeating periods are estimated as the indices of the maximum values
    P = P+r[0]
    P = P.astype(int)
    return P


"""
Repeating mask from the magnitude spectrogram and the repeating periods
M = repeating_mask(V,p,k);

Input(s):
V: magnitude spectrogram [n bins, m frames]
p: repeating periods in time frames [1, m frames]
k: order for the median filter

Output(s):
M: repeating (soft) mask in [0,1] [n bins, m frames]
"""
def repeating_mask(V,p,k):
    # Number of frequency bins and time frames
    n,m = V.shape
    # Order vector centered in 0
    k = np.arange(1,k+1)-int(np.ceil(k/2.))
    W = np.zeros((n,m))
    # Loop over the frames
    for j in range(int(m)):
    	  # Indices of the frames for the median filtering  (e.g.: k=3 => i=[-1,0,1], k=4 => i=[-1,0,1,2])
        i = j+k*p[j]
        # Discard out-of-range indices
        i = i[i>=0]
        i = i[i<m]
        # Median filter centered on frame j
        W[:,j] = np.median(np.real(V[:,i.astype(int)]),1)
    # For every time-frequency bins, we must have W <= V    
    W = np.minimum(V,W)
    # Normalize W by V
    eps = float_info.epsilon
    M = (W+eps)/(V+eps)
    return M

def parse_input_files(input_files, ext='.wav'):
    """
    Collect all files by given extension and keywords.

    :param agrs:  class 'argparse.Namespace'.
    :param ext:   the string of file extension.
    :returns:     a list of stings of file name.

    """
    from os.path import basename, isdir
    import fnmatch
    import glob
    files = []

    # check what we have (file/path)
    if isdir(input_files):
        # use all files with .raw.melody in the given path
        files = fnmatch.filter(glob.glob(input_files+'/*'), '*'+ext)
    else:
        # file was given, append to list
        if basename(input_files).find(ext)!=-1:
            files.append(input_files)
    print '  Input files: '
    for f in files: print '    ', f
    return files

def parser():
    """
    Parses the command line arguments.

    :param lgd:       use local group delay weighting by default
    :param threshold: default value for threshold

    """
    import argparse
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    REpeating Pattern Extraction Technique (REPET): adaptive REPET (thanks Antoine!)
     
    REPET is a simple method for separating the repeating background 
    from the non-repeating foreground in an audio mixture. 
    REPET can be extended by locally modeling the repetitions.
     
    Usage:
        y = repet_ada(x,fs,per,par);

    Input(s):
        x: mixture data [t samples, k channels]
        fs: sampling frequency in Hz
        per: repeating period range (if two values) 
             or defined repeating period (if one value) in seconds 
             (default: [0.8,min(8,seg(1)/3)])
        par: adaptive parameters (two values) (default: [24,12,7])
             par(1): adaptive window length in seconds
             par(2): adaptive step length in seconds
             par(3): order for the median filter

    Output(s):
        y: repeating background [t samples, k channels]
           (the corresponding non-repeating foreground is equal to x-y)

    Example(s):
        # Read some audio mixture
        x, fs, nbits = scipy.io.wavfile.read('mixture.wav')
        # Derives the repeating background using windows of 24 seconds, steps of 12 seconds, and order of 7
        y = repet(x,fs,[0.8,8],[24,12,7])
        # Write the repeating background
        wavwrite(y,fs,nbits,'background.wav')
        # Write the corresponding non-repeating foreground
        wavwrite(x-y,fs,nbits,'foreground.wav')

    See also http://music.eecs.northwestern.edu/research.php?project=repet

    Author: Zafar Rafii (zafarrafii@u.northwestern.edu)
    Update: September 2013
    Copyright: Zafar Rafii and Bryan Pardo, Northwestern University
    Reference(s):
        [1]: Antoine Liutkus, Zafar Rafii, Roland Badeau, Bryan Pardo, and Gaël Richard. 
             "Adaptive Filtering for Music/Voice Separation Exploiting the Repeating Musical Structure," 
             37th International Conference on Acoustics, Speech and Signal Processing,
             Kyoto, Japan, March 25-30, 2012.

    """)
    # general options
    p.add_argument('input_files', type=str, metavar='input_files', nargs='+',
                   help='files to be processed')
    p.add_argument('output_dir', type=str, metavar='output_dir',
                   help='output directory.')
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2015-08-18)')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return args
    return args

def main(args):
    """
    Main adaptive Repeating Pattern Extraction Technique program.

    :param args: parsed arguments

    """
    print 'Running monaural source separation...'
    
    # parse and list files to be processed
    files = parse_input_files(args)
    
    # create result directory
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    print '  Output directory: ', '\n', '    ', args.output_dir

    # processing
    for f in files:
        # parse file name and extension
        ext = os.path.basename(f).split('.')[-1]
        name = os.path.basename(f).split('.')[0]    
        
        # do the processing stuff 
        fs, x = wavfile.read(f)
        # change data type int to float and normalization
        x = x.astype(np.float)/np.max(x)
        # execute main adaptive REPET function
        y = repet_ada(x,fs)
        z = x-y
        z = z/(np.max(z)/2**15)
        z = z.astype(np.int16)
        wavfile.write(args.output_dir+os.sep+name+'_sep.wav',fs,z)


if __name__ == '__main__':
    args = parser()
    main(args)
