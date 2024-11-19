import os
import numpy as np
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score as bal_acc

def select_contacts(channels, nframes, elec_map, PTD_map, category):
    names = list(channels)*nframes
    indices = []
    excluded = []
    for i, chan in enumerate(names):
        anat  = elec_map.get(chan, 'NoValue')
        val = PTD_map.get(chan, 'NoValue')
        if anat == 'NoValue':
            if chan not in excluded:
                # logging.warning(f'{pt} | Channel {chan} is not in electrode locations and has been excluded')
                excluded.append(chan)
        elif anat == 'Unknown':
            if chan not in excluded:
                # logging.warning(f'{pt} | Channel {chan} location is unknown and has been excluded')
                excluded.append(chan)    
        if anat[6:15] == '_G_and_S_' or 'G_Ins_lg_and_S_cent_ins' in anat:
            if category == 'GS':
                indices.append(i)
        elif anat[6:9] == '_G_' or 'Pole_' in anat:
            if category == 'gyrus':
                indices.append(i)
        elif anat[6:9] == '_S_' or 'Lat_' in anat:
            if category == 'sulcus':
                indices.append(i)
        if val != 'NoValue':
            if val < 0:
                if category == 'WM':
                    indices.append(i)
            elif val > 0:
                if category == 'GM':
                    indices.append(i) 
        if 'Left-' in anat or 'Right-' in anat:
            if '-White-' not in anat:
                if category == 'SC':
                    indices.append(i) 
        elif 'ctx_' in anat:
            if category == 'cortical':
                indices.append(i)
    return indices

if __name__=="__main__":
    
    scale = 'global'
    categories = ['GS'] #['all', 'GM', 'WM', 'cortical', 'SC', 'gyrus', 'sulcus', 'GS']

    data_path = r'./Shared/Data/'
    feat_path = r'./Shared/Results/Preprocessed/'
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
            channel_names = np.load(f'{feat_path}/{pt}_channels.npy')   
            labels = np.load(f'{feat_path}/{pt}_labels.npy')

            #Load electrode locations
            elec_locs = read_csv(f'{data_path}/{pt}_electrode_locations.csv')
            elec_map = dict(zip(elec_locs['electrode_name_1'], elec_locs['location']))
            PTD_map = dict(zip(elec_locs['electrode_name_1'], elec_locs['PTD']))

            #Exclude future (and early past) features
            if exclFuture:
                cutin = int(eeg.shape[1]/nframes*6) #take from T-4
                cutoff = int(eeg.shape[1]/nframes*11) #keep until T1 (5 timeframes)
                data = eeg[:,cutin:cutoff]
            
            #Select contacts depending on category
            if category != 'all':
                indices = select_contacts(channel_names, 5, elec_map, PTD_map, category)
                channels = channel_names[indices[:int(len(indices)/5)]]
                eeg = data[:,indices]
            else:
                channels = channel_names
                eeg = data

            print(f'{pt} | {len(channels)} channels in category {category}')

            if len(channels) < 1:
                print(f'{pt} | No channels for this category - participant will get empty arrays')
                predictions = np.empty((0,len(labels)))
                balacc = np.empty((0,len(labels)))
                #Save target, prediction, accuracy and included channels
                os.makedirs(os.path.join(result_path), exist_ok=True)
                np.save(f'{result_path}/{pt}_target.npy', labels)
                np.save(f'{result_path}/{pt}_predictions.npy', predictions)
                np.save(f'{result_path}/{pt}_accuracy.npy', balacc)
                np.save(f'{result_path}/{pt}_channels.npy', channels)
                continue 

            #Initialize arrays to save the predictions and balanced accuracy
            predictions = np.empty(len(labels))
            balacc = np.zeros(nfolds)

            for k,(train, test) in enumerate(kf.split(eeg)):
                #Z-Normalize with mean and std from the training data
                mu=np.mean(eeg[train,:],axis=0)
                std=np.std(eeg[train,:],axis=0)
                trainData=(eeg[train,:]-mu)/std
                testData=(eeg[test,:]-mu)/std
                trainLabel = labels[train]
                testLabel = labels[test]
                #Fit the model
                est.fit(trainData, trainLabel)
                #Predict for the test data
                predict_fold = est.predict(testData)
                #Save the fold
                predictions[test] = predict_fold
                #Balanced accuracy
                balacc[k] = bal_acc(testLabel, predict_fold)

            #Save target, prediction, accuracy and included channels
            os.makedirs(result_path, exist_ok=True)
            np.save(f'{result_path}/{pt}_target.npy', labels)
            np.save(f'{result_path}/{pt}_predictions.npy', predictions)
            np.save(f'{result_path}/{pt}_accuracy.npy', balacc)
            np.save(f'{result_path}/{pt}_channels.npy', channels)

            print(f'{pt} | Scale: {scale} | Category: {category} | Finished')

print('All done!')

