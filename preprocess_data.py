import os
import numpy as np 
import numpy.matlib as matlib
from mne.filter import filter_data
import scipy
from scipy import fftpack
from scipy.stats import mode
from scipy.signal import decimate, hilbert
from sklearn.preprocessing import LabelEncoder
import MelFilterBank as mel

#Helper function to drastically speed up the hilbert transform of larger data
hilbert3 = lambda x: hilbert(x, fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def cleanData(data, channels):
    """
    Clean data by removing irrelevant channels
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    channels: array (electrodes, label)
        Labels of the electrodes
    
    Returns
    ----------
    data: array (samples, channels)
        Cleaned EEG time series
    channels: array (electrodes, label)
        Cleaned labels of the electrodes      
    """
    clean_data = []
    clean_channels = []
    for i in range(channels.shape[0]):
        if '+' in channels[i][0]: #EKG/MRK/etc channels
            continue
        elif channels[i][0][0] == 'E': #Empty channels
            continue
        elif channels[i][0][:2] == 'el': #Empty channels
            continue
        elif channels[i][0][0] in ['F','C','T','O','P']: #Other channels
            continue        
        else:
            clean_channels.append(channels[i])
            clean_data.append(data[:,i])
    return np.transpose(np.array(clean_data,dtype="float64")), np.array(clean_channels)

    """Apply a common-average re-reference to the data
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    
    Returns
    ----------
    data: array (samples, channels)
        CAR re-referenced EEG time series   
    """
    data_CAR = np.zeros((data.shape[0], data.shape[1]))
    average = np.average(data, axis=1)
    for i in range(data.shape[1]):
        data_CAR[:,i] = data[:,i] - average
    return data_CAR

def elecShaftR(data, channels):
    """
    Apply an electrode-shaft re-reference to the data
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    channels: array (electrodes, label)
        Channel names
    
    Returns
    ----------
    data: array (samples, channels)
        ESR re-referenced EEG time series   
    """
    data_ESR = np.zeros((data.shape[0], data.shape[1]))
    #Get shaft information
    shafts = {}
    for i,chan in enumerate(channels):
        if chan[0].rstrip('0123456789') not in shafts:
            shafts[chan[0].rstrip('0123456789')] = {'start': i, 'size': 1}
        else:
            shafts[chan[0].rstrip('0123456789')]['size'] += 1
    #Get average signal per shaft
    for shaft in shafts:
        shafts[shaft]['average'] = np.average(data[:,shafts[shaft]['start']:(shafts[shaft]['start']+shafts[shaft]['size'])], axis=1)
    #Subtract the shaft average from each respective channel   
    for i in range(data.shape[1]):
        data_ESR[:,i] = data[:,i] - shafts[channels[i][0].rstrip('0123456789')]['average']
    return data_ESR

def extractFB(data, sr, windowLength=0.05, frameshift=0.01):
    """
    Window data and extract frequency-band envelope using the hilbert transform
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    
    Returns
    ----------
    feat, array shape (windows, channels)
        Frequency-band feature matrix
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    numWindows=int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    #Band-pass for high-frequencies (between 70 and 170 Hz)
    data = filter_data(data.T, sr, 70,170,method='iir').T 
    #Band-stop filter for first two harmonics of 50 Hz line noise
    data = filter_data(data.T, sr, 102, 98,method='iir').T 
    data = filter_data(data.T, sr, 152, 148,method='iir').T
    data = np.abs(hilbert3(data))
    feat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:] = np.mean(data[start:stop,:],axis=0)
    return feat

def stackFeatures(features, modelOrder=4, stepSize=5):
    """
    Add temporal context to each window by stacking neighboring feature vectors
    
    Parameters
    ----------
    features: array (windows, channels)
        Feature time series
    modelOrder: int
        Number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate for frameshift)
    
    Returns
    ----------
    featStacked, array shape (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    featStacked=np.zeros((features.shape[0]-(2*modelOrder*stepSize),(2*modelOrder+1)*features.shape[1]))
    for fNum,i in enumerate(range(modelOrder*stepSize,features.shape[0]-modelOrder*stepSize)):
        ef=features[i-modelOrder*stepSize:i+modelOrder*stepSize+1:stepSize,:]
        featStacked[fNum,:]=ef.flatten() # Add 'F' if stacked the same as matlab
    return featStacked

def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01,numFilter=23):
    """
    Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms
    
    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    numFilter: int
        Number of triangular filters in the mel filterbank
    
    Returns
    ----------
    spectrogram, array shape (numWindows, numFilter)
        Logarithmic mel scaled spectrogram
    """
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    win = np.hanning(np.floor(windowLength*sr + 1))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(windowLength*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        startAudio = int(np.floor((w*frameshift)*sr))
        stopAudio = int(np.floor(startAudio+windowLength*sr))
        a = audio[startAudio:stopAudio]
        spec = np.fft.rfft(win*a)
        spectrogram[w,:] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], numFilter, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram

def nameVector(elecs, modelOrder=4):
    """
    Creates list of electrode names
    
    Parameters
    ----------
    elecs: array of strings 
        Original electrode names
    modelOrder: int
        Temporal context stacked prior and after current window
        Will be added as T-modelOrder, T-(modelOrder+1), ...,  T0, ..., T+modelOrder
        to the elctrode names

    Returns
    ----------
    names: array of strings 
        List of electrodes including contexts, will have size elecs.shape[0]*(2*modelOrder+1)
    """
    names = matlib.repmat(elecs.astype(np.dtype(('U', 10))),1,2 * modelOrder +1).T
    for i, off in enumerate(range(-modelOrder,modelOrder+1)):
        names[i,:] = [e[0] + 'T' + str(off) for e in elecs]
    return names.flatten()  # Add 'F' if stacked the same as matlab

def labelSpeech(melspec):
    spec_avg = np.mean(melspec, axis=1)
    threshold = (np.max(spec_avg)+np.min(spec_avg))*0.45
    labels = np.where(spec_avg>threshold, 'Speech', 'Silence')
    return labels

if __name__=="__main__":

    data_path = r'./Shared/Data/'
    result_path = r'./Shared/Results/Preprocessed/'
    pt_ids = ['P%02d'%i for i in range(1,31)]

    winL = 0.05
    frameshift = 0.01
    modelOrder = 10
    stepSize = 5
    sr = 1024
    ref = 'ESR'
    band = 'HFA'
  
    for pti, pt in enumerate(pt_ids):

        print(f'{pt} | Preprocessing | Running')

        #Load data
        data = np.load(f'{data_path}/{pt}_sEEG.npy')
        elecs = np.load(f'{data_path}/{pt}_channels.npy')
        words = np.load(f'{data_path}/{pt}_stimuli.npy')
        audio = np.load(f'{data_path}/{pt}_audio.npy')

        #Clean up irrelevant channels
        data, channels = cleanData(data,elecs)
        
        #Re-referencing
        if ref == 'ESR':
            data = elecShaftR(data, channels) 
            
        #Extract features
        feat = extractFB(data,sr,windowLength=winL,frameshift=frameshift)

        #Process audio
        #downsample audio to 16kHz
        audioSR = 48000
        targetSR = 16000
        audio = decimate(audio,int(audioSR / targetSR))
        #scale the audio
        scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)  
        #extact log mel-scaled spectrograms
        spectrogram = extractMelSpecs(scaled,targetSR,windowLength=winL,frameshift=frameshift,numFilter=23)

        #Extract labels
        labels = labelSpeech(spectrogram)
        le = LabelEncoder().fit(labels)
        target = le.transform(labels)

        #Stack features
        feat = stackFeatures(feat,modelOrder=modelOrder,stepSize=stepSize)

        #Align to sEEG features
        target = target[modelOrder*stepSize:target.shape[0]-modelOrder*stepSize]
        spectrogram = spectrogram[modelOrder*stepSize:spectrogram.shape[0]-modelOrder*stepSize,:]
        #adjust length (differences might occur due to rounding in the number of windows)
        if spectrogram.shape[0]!=feat.shape[0]:
            tLen = np.min([spectrogram.shape[0],feat.shape[0]])
            spectrogram = spectrogram[:tLen,:]
            feat = feat[:tLen,:]
            target = target[:tLen]            
        
        #Create feature names by appending the temporal shift 
        feat_names = nameVector(channels, modelOrder=modelOrder)

        #Save preprocessed data
        os.makedirs(result_path, exist_ok=True)
        np.save(f'{result_path}/{pt}_features.npy', feat)
        np.save(f'{result_path}/{pt}_channels.npy', channels.flatten())   
        np.save(f'{result_path}/{pt}_feat_names.npy', feat_names)
        np.save(f'{result_path}/{pt}_spectrogram.npy', spectrogram)
        np.save(f'{result_path}/{pt}_labels.npy', target)
        
        print(f'{pt} | Preprocessing | Finished')

print('All done!')          
