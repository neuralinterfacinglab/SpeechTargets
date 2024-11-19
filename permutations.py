import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score as bal_acc

def calculate_significance_threshold(pt_ids, permutations):
    sign_thresholds = []
    for pti, pt in enumerate(pt_ids):
        folds = []
        for k in range(permutations[pti].shape[1]):
            folds.append(np.percentile(permutations[pti][:,k],99))
        sign_thresholds.append(np.max(folds))
    sign_threshold = np.max(sign_thresholds)
    print(f'Significance threshold: {sign_threshold:.3f}')
    return sign_threshold


if __name__=="__main__":

    #Settings
    feat_path = r'./Shared/Results/Preprocessed'
    result_path = r'./Shared/Results/Permutations'
    pt_ids = ['P%02d'%i for i in range(1,31)]

    nfolds = 10
    numRands = 1000
    
    kf = KFold(nfolds,shuffle=False)

    #Initialize array
    randomControl = np.zeros((len(pt_ids),numRands,nfolds))
    
    # Run permutations #
    for pti, pt in enumerate(pt_ids):

        #Load target labels
        target = np.load(f'{feat_path}/{pt}_labels.npy')

        #Estimate random baseline
        for k,(train, test) in enumerate(kf.split(target)):
            data = target[test]
            order = np.arange(data.shape[0])
            for randRound in range(numRands):
                np.random.shuffle(order)
                shuffled = data[order]
                randomControl[pti, randRound, k] = bal_acc(data, shuffled)
        
        #Save the permutations per participant
        os.makedirs(result_path, exist_ok=True)
        np.save(f'{result_path}/{pt}_permutations.npy', randomControl[pti])

    # Calculate significance threshold #
    sign_threshold = calculate_significance_threshold(pt_ids, randomControl)

print("All done!")

