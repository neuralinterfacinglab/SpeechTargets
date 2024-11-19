import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score as bal_acc

def get_channel_indices(channel_names, selected_channels, timeframes=5):
    """
    Get name of each channel and its indices

    Parameters
    ----------
    channel_names: array (electrodes, label)
        Channel names
    
    Returns
    ----------
    channel_indices: dict (channel name, indices)
        Channel names with corresponding indices  
    """
    #multiply for timeframes
    names = list(channel_names)*timeframes
    #get channel information
    channel_indices = {}
    for i,chan in enumerate(names):
        if chan in selected_channels:
            if chan not in channel_indices:
                channel_indices[chan] = [i]
            else:
                channel_indices[chan].append(i)
    return channel_indices

if __name__=="__main__":
    
    scale = 'local'
    category = 'all' #only need to run all - divide in categories afterwards

    data_path = r'./Shared/Data/'
    feat_path = r'./Shared/Results/Preprocessed/'
    result_path = fr'./Shared/Results/LDA/{scale}/{category}'
    pt_ids = ['P%02d'%i for i in range(1,31)]

    winLength = 0.05
    frameshift = 0.01
    audiosr = 16000
    nframes = 21 #number of timeframes
    nfolds = 10 #number of folds for cross-validation
  
    exclFuture = True #exclude future timeframes

    kf = KFold(nfolds,shuffle=False)
    est = LinearDiscriminantAnalysis()

    for pti, pt in enumerate(pt_ids):

        print(f'{pt} | Scale: {scale} | Category: {category} | Running')

        #Load data
        eeg = np.load(f'{feat_path}/{pt}_features.npy')
        channel_names = np.load(f'{feat_path}/{pt}_channels.npy')   
        labels = np.load(f'{feat_path}/{pt}_labels.npy')

        #Exclude future (and early past) features
        if exclFuture:
            cutin = int(eeg.shape[1]/nframes*6) #take from T-4
            cutoff = int(eeg.shape[1]/nframes*11) #keep until T1 (5 timeframes)
            data = eeg[:,cutin:cutoff]
        
        #Get channel information (indices)
        selected_channels = channel_names
        channels = get_channel_indices(channel_names, selected_channels, timeframes=5)

        #Initialize arrays to save the predictions and balanced accuracy
        predictions = np.empty((len(channels),len(labels)))
        balacc = np.zeros((len(channels),nfolds))

        print(f'{pt} | {len(channels)} channels in total')

        #Do a cross-validation across all channels
        for ch, channel in enumerate(channels):

            #Select the features of this channel
            feats = data[:,channels[channel]]

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
                predictions[ch,test] = predict_fold
                #Balanced accuracy
                balacc[ch,k] = bal_acc(testLabel, predict_fold)

        #Save target, prediction, accuracy and included channels
        os.makedirs(result_path, exist_ok=True)
        np.save(f'{result_path}/{pt}_target.npy', labels)
        np.save(f'{result_path}/{pt}_predictions.npy', predictions)
        np.save(f'{result_path}/{pt}_accuracy.npy', balacc)
        np.save(f'{result_path}/{pt}_channels.npy', channel_names)

        print(f'{pt} | Scale: {scale} | Category: {category} | Finished')

print('All done!')

