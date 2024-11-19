import os
import numpy as np
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score as bal_acc

def get_shaft_indices(channels, timeframes=5):
    """
    Get name of each shaft and its indices

    Parameters
    ----------
    channels: array (electrodes, label)
        Channel names
    
    Returns
    ----------
    shafts: dict (shaft name, indices)
        Shaft names with corresponding indices  
    """
    #multiply for timeframes
    names = list(channels)*timeframes
    #get shaft information
    shafts = {}
    for i,chan in enumerate(names):
        if chan.rstrip('0123456789') not in shafts:
            shafts[chan.rstrip('0123456789')] = [i]
        else:
            shafts[chan.rstrip('0123456789')].append(i)
    return shafts

def get_shaft_indices_GM(pt, channels, PTD_map, timeframes=5):
    """
    Get name of each shaft and its indices

    Parameters
    ----------
    channels: array (electrodes, label)
        Channel names
    PTD_map: dict(elec label: PTD value)
    
    Returns
    ----------
    shafts: dict (shaft name, indices)
        Shaft names with corresponding indices  
    """
    #multiply for timeframes
    names = list(channels)*timeframes
    #get shaft information
    shafts = {}
    excluded = []
    for i,chan in enumerate(names):
        val  = PTD_map.get(chan, 'NoValue')
        if val == 'NoValue':
            if chan not in excluded:
                print(f'{pt} | Channel {chan} is not in electrode locations and has been excluded')
                excluded.append(chan)
            continue
        if val > 0: #anything above 0 has more gray matter
            if chan.rstrip('0123456789') not in shafts:
                shafts[chan.rstrip('0123456789')] = [i]
            else:
                shafts[chan.rstrip('0123456789')].append(i)
        elif val == 0:
            if chan not in excluded:
                print(f'{pt} | Channel {chan} has a PTD of 0 and has been excluded')
                excluded.append(chan)
        elif np.isnan(val):
            if chan not in excluded:
                print(f'{pt} | Channel {chan} has a PTD of NaN and has been excluded')  
                excluded.append(chan)  
    return shafts

def get_shaft_indices_WM(pt, channels, PTD_map, timeframes=5):
    """
    Get name of each shaft and its indices, excluding

    Parameters
    ----------
    channels: array (electrodes, label)
        Channel names
    PTD_map: dict(elec label: PTD value)
    
    Returns
    ----------
    shafts: dict (shaft name, indices)
        Shaft names with corresponding indices  
    """
    #multiply for timeframes
    names = list(channels)*timeframes
    #get shaft information
    shafts = {}
    excluded = []
    for i,chan in enumerate(names):
        val  = PTD_map.get(chan, 'NoValue')
        if val == 'NoValue':
            if chan not in excluded:
                print(f'{pt} | Channel {chan} is not in electrode locations and has been excluded')
                excluded.append(chan)
            continue
        if val < 0: #anything below 0 has more white matter
            if chan.rstrip('0123456789') not in shafts:
                shafts[chan.rstrip('0123456789')] = [i]
            else:
                shafts[chan.rstrip('0123456789')].append(i)
        elif val == 0:
            if chan not in excluded:
                print(f'{pt} | Channel {chan} has a PTD of 0 and has been excluded')
                excluded.append(chan)
        elif np.isnan(val):
            if chan not in excluded:
                print(f'{pt} | Channel {chan} has a PTD of NaN and has been excluded')
                excluded.append(chan)
    return shafts

def get_shaft_indices_cortical(pt, channels, elec_map, timeframes=5):
    """
    Get name of each shaft and its indices

    Parameters
    ----------
    channels: array (electrodes, label)
        Channel names
    elec_map: dict(elec label: anatomy label)
    
    Returns
    ----------
    shafts: dict (shaft name, indices)
        Shaft names with corresponding indices  
    """
    #multiply for timeframes
    names = list(channels)*timeframes
    #get shaft information
    shafts = {}
    excluded = []
    for i, chan in enumerate(names):
        anat  = elec_map.get(chan, 'NoValue')
        if anat == 'NoValue':
            if chan not in excluded:
                print(f'{pt} | Channel {chan} is not in electrode locations and has been excluded')
                excluded.append(chan)
        elif anat == 'Unknown':
            if chan not in excluded:
                print(f'{pt} | Channel {chan} location is unknown and has been excluded')
                excluded.append(chan)
        elif anat[0:4] == 'ctx_': #This is it
            if chan.rstrip('0123456789') not in shafts:
                shafts[chan.rstrip('0123456789')] = [i]
            else:
                shafts[chan.rstrip('0123456789')].append(i)
    return shafts

def get_shaft_indices_SC(channels, elec_map, timeframes=5):
    """
    Get name of each shaft and its indices

    Parameters
    ----------
    channels: array (electrodes, label)
        Channel names
    elec_map: dict(elec label: anatomy label)
    
    Returns
    ----------
    shafts: dict (shaft name, indices)
        Shaft names with corresponding indices  
    """
    #multiply for timeframes
    names = list(channels)*timeframes
    #get shaft information
    shafts = {}
    excluded = []
    for i, chan in enumerate(names):
        anat  = elec_map.get(chan, 'NoValue')
        if anat == 'NoValue':
            if chan not in excluded:
                excluded.append(chan)
        elif anat == 'Unknown':
            if chan not in excluded:
                excluded.append(chan)
        elif 'Left-' in anat or 'Right-' in anat: #This is it
            if '-White-' not in anat: #Make sure it isn't white matter
                if chan.rstrip('0123456789') not in shafts:
                    shafts[chan.rstrip('0123456789')] = [i]
                else:
                    shafts[chan.rstrip('0123456789')].append(i)
    return shafts

def get_shaft_indices_gyrus(channels, elec_map, timeframes=5):
    """
    Get name of each shaft and its indices, excluding

    Parameters
    ----------
    channels: array (electrodes, label)
        Channel names
    elec_map: dict(elec label: anatomy label)
    
    Returns
    ----------
    shafts: dict (shaft name, indices)
        Shaft names with corresponding indices  
    """
    #multiply for timeframes
    names = list(channels)*timeframes
    #get shaft information
    shafts = {}
    excluded = []
    for i, chan in enumerate(names):
        anat  = elec_map.get(chan, 'NoValue')
        if anat == 'NoValue':
            if chan not in excluded:
                excluded.append(chan)
        elif anat == 'Unknown':
            if chan not in excluded:
                excluded.append(chan)
        elif anat[6:9] == '_G_' or 'Pole_' in anat: #This is it
            if anat[6:15] != '_G_and_S_': #Make sure it is not on the border
                if chan.rstrip('0123456789') not in shafts:
                    shafts[chan.rstrip('0123456789')] = [i]
                else:
                    shafts[chan.rstrip('0123456789')].append(i)
    return shafts

def get_shaft_indices_sulcus(channels, elec_map, timeframes=5):
    """
    Get name of each shaft and its indices, excluding

    Parameters
    ----------
    channels: array (electrodes, label)
        Channel names
    elec_map: dict(elec label: anatomy label)
    
    Returns
    ----------
    shafts: dict (shaft name, indices)
        Shaft names with corresponding indices  
    """
    #multiply for timeframes
    names = list(channels)*timeframes
    #get shaft information
    shafts = {}
    excluded = []
    for i, chan in enumerate(names):
        anat  = elec_map.get(chan, 'NoValue')
        if anat == 'NoValue':
            if chan not in excluded:
                excluded.append(chan)
        elif anat == 'Unknown':
            if chan not in excluded:
                excluded.append(chan)
        elif anat[6:9] == '_S_' or 'Lat_' in anat: #This is it
            if chan.rstrip('0123456789') not in shafts:
                shafts[chan.rstrip('0123456789')] = [i]
            else:
                shafts[chan.rstrip('0123456789')].append(i)
    return shafts

def get_shaft_indices_GS(channels, elec_map, timeframes=5):
    """Get name of each shaft and its indices, excluding

    Parameters
    ----------
    channels: array (electrodes, label)
        Channel names
    elec_map: dict(elec label: anatomy label)
    
    Returns
    ----------
    shafts: dict (shaft name, indices)
        Shaft names with corresponding indices  
    """
    #multiply for timeframes
    names = list(channels)*timeframes
    #get shaft information
    shafts = {}
    excluded = []
    for i, chan in enumerate(names):
        anat  = elec_map.get(chan, 'NoValue')
        if anat == 'NoValue':
            if chan not in excluded:
                excluded.append(chan)
        elif anat == 'Unknown':
            if chan not in excluded:
                excluded.append(chan)
        elif anat[6:15] == '_G_and_S_' or 'G_Ins_lg_and_S_cent_ins' in anat: #This is it
            if chan.rstrip('0123456789') not in shafts:
                shafts[chan.rstrip('0123456789')] = [i]
            else:
                shafts[chan.rstrip('0123456789')].append(i)
    return shafts

if __name__=="__main__":
    
    scale = 'shaft'
    categories = ['all', 'GM', 'WM', 'cortical', 'SC', 'gyrus', 'sulcus', 'GS']

    data_path = fr'./Shared/Data/'
    feat_path = fr'./Shared/Results/Preprocessed/'
    pt_ids = ['P%02d'%i for i in range(1,31)]

    winLength = 0.05
    frameshift = 0.01
    audiosr = 16000
    nframes = 21 #number of timeframes
    nfolds = 10 #number of folds for cross-validation
  
    exclFuture = True #exclude future timeframes

    kf = KFold(nfolds,shuffle=False)
    est = LinearDiscriminantAnalysis()

    for category in categories:

        result_path = fr'./Shared/Results/LDA/{scale}/{category}'

        for pti, pt in enumerate(pt_ids):

            print(f'{pt} | Scale: {scale} | Category: {category} | Running')

            #Load data
            eeg = np.load(f'{feat_path}/{pt}_features.npy')
            channels = np.load(f'{feat_path}/{pt}_channels.npy')   
            labels = np.load(f'{feat_path}/{pt}_labels.npy')

            #Load electrodes locations
            elec_locs = read_csv(f'{data_path}/{pt}_electrode_locations.csv')
            elec_map = dict(zip(elec_locs['electrode_name_1'], elec_locs['location']))
            PTD_map = dict(zip(elec_locs['electrode_name_1'], elec_locs['PTD']))

            #Exclude future (and early past) features
            if exclFuture:
                cutin = int(eeg.shape[1]/nframes*6) #take from T-4
                cutoff = int(eeg.shape[1]/nframes*11) #keep until T1 (5 timeframes)
                data = eeg[:,cutin:cutoff]
            
            #Get shaft information
            if category == 'all':
                shafts = get_shaft_indices(channels)
            elif category == 'GM':
                shafts = get_shaft_indices_GM(pt, channels, PTD_map)
            elif category == 'WM':
                shafts = get_shaft_indices_WM(pt, channels, PTD_map)
            elif category == 'cortical':
                shafts = get_shaft_indices_cortical(pt, channels, elec_map)
            elif category == 'SC':
                shafts = get_shaft_indices_SC(channels, elec_map)                
            elif category == 'gyrus':
                shafts = get_shaft_indices_gyrus(channels, elec_map)
            elif category == 'sulcus':
                shafts = get_shaft_indices_sulcus(channels, elec_map)
            elif category == 'GS':
                shafts = get_shaft_indices_GS(channels, elec_map)

            #Initialize arrays to save the predictions and balanced accuracy
            predictions = np.empty((len(shafts),len(labels)))
            balacc = np.zeros((len(shafts),nfolds))

            print(f'{pt} | {len(shafts)} shafts in total')

            #Do a cross-validation across all shafts
            for s, shaft in enumerate(shafts):

                #Select the features within this shaft
                feats = data[:,shafts[shaft]]

                for k,(train, test) in enumerate(kf.split(feats)):
                    #Z-Normalize with mean and std from the training data
                    mu=np.mean(feats[train,:],axis=0)
                    std=np.std(feats[train,:],axis=0)
                    trainData=(feats[train,:]-mu)/std
                    testData=(feats[test,:]-mu)/std
                    trainLabel = labels[train]
                    testLabel = labels[test]
                    #Fit the model
                    est.fit(trainData, trainLabel)
                    #Predict for the test data
                    predict_fold = est.predict(testData)
                    #Save the fold
                    predictions[s,test] = predict_fold
                    #Balanced accuracy
                    balacc[s,k] = bal_acc(testLabel, predict_fold)

            #Get included shaft and channel names
            shaft_list = []
            channels_included = []
            for shaft in shafts:
                shaft_list.append(shaft)
                for i,ch in enumerate(shafts[shaft]):
                    if ch < len(channels):
                        channels_included.append(channels[ch])

            #Save target, prediction, accuracy and included channels
            os.makedirs(result_path, exist_ok=True)
            np.save(f'{result_path}/{pt}_target.npy', labels)
            np.save(f'{result_path}/{pt}_predictions.npy', predictions)
            np.save(f'{result_path}/{pt}_accuracy.npy', balacc)
            np.save(f'{result_path}/{pt}_shafts.npy', shaft_list)
            np.save(f'{result_path}/{pt}_channels.npy', channels_included)

            print(f'{pt} | Scale: {scale} | Category: {category} | Finished')

print('All done!')

