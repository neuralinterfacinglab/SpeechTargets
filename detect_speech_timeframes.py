import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score as bal_acc

if __name__=="__main__":
    
    analysis = 'timeframes'
    category = 'all'

    data_path = r'./Shared/Data/'
    feat_path = r'./Shared/Results/Preprocessed/'
    result_path = fr'.Shared/Results/LDA/{analysis}/{category}'
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

        print(f'{pt} | Timeframes | Running')

        #Load data
        eeg = np.load(f'{feat_path}/{pt}_features.npy')
        channels = np.load(f'{feat_path}/{pt}_channels.npy')   
        labels = np.load(f'{feat_path}/{pt}_labels.npy')

        print(f'{pt} | {eeg.shape[1]} features in total')

        #Initialize arrays to save the predictions and balanced accuracy
        predictions = np.empty((nframes,len(channels),len(labels)))
        balacc = np.zeros((nframes,len(channels),nfolds))
        
        for tf in range(0,nframes):

            #Extract data from 1 timeframe
            cuton = int(eeg.shape[1]/nframes*(tf))
            cutoff = int(eeg.shape[1]/nframes*(tf+1))
            data = eeg[:,cuton:cutoff]

            #Do a cross-validation across all channels
            for ch, channel in enumerate(channels):

                #Select the feature
                feats = data[:,ch]

                for k,(train, test) in enumerate(kf.split(feats)):
                    #Z-Normalize with mean and std from the training data
                    mu=np.mean(feats[train],axis=0)
                    std=np.std(feats[train],axis=0)
                    trainData=(feats[train]-mu)/std
                    testData=(feats[test]-mu)/std
                    trainLabel = labels[train]
                    testLabel = labels[test]
                    #Fit the model
                    est.fit(trainData.reshape(-1,1), trainLabel)
                    #Predict for the test data
                    predict_fold = est.predict(testData.reshape(-1,1))
                    #Save the fold
                    predictions[tf,ch,test] = predict_fold
                    #Balanced accuracy
                    balacc[tf,ch,k] = bal_acc(testLabel, predict_fold)
                    
        #Save target, prediction, accuracy and included channels
        os.makedirs(result_path, exist_ok=True)
        np.save(f'{result_path}/{pt}_target.npy', labels)
        np.save(f'{result_path}/{pt}_predictions.npy', predictions)
        np.save(f'{result_path}/{pt}_accuracy.npy', balacc)
        np.save(f'{result_path}/{pt}_channels.npy', channels)

        print(f'{pt} | Timeframes | Finished')       

print('All done!')

