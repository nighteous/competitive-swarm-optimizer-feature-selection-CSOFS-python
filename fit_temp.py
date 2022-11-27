import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import utils

"""
Implementation of func.m for calculating error for fitness
"""

def func(data, inp):

    
    # if inp length not 0 then return 1 
    if len(inp) == 0:
        return 1

    k = 10
    group = data[:, -1]
    classes = np.unique(group)
    data_group = []
    smile_subsample_segments = []

    data[:,:-1] = (data[:,:-1] - data[:,:-1].mean()) / data[:,:-1].std() # standardizing all rows until -1
    
    for i in range(len(classes)):

        sa = []
        sa = np.array([data[ind] for ind in range(len(group)) if group[ind] == classes[i]])
        
        number_of_smile_samples, _ = sa.shape
        smile_temp = np.round(np.linspace(1, number_of_smile_samples, k + 1))
        smile_temp = [i - 1 for i in smile_temp]

        data_group.append(sa)
        smile_subsample_segments.append(smile_temp)

    fit_temp = np.zeros((1, 10))

    mdl = KNeighborsClassifier(n_neighbors=4)

    for i in range(k):
        data_tr = None
        data_ts = None

        for j in range(len(classes)):
            smile_temp = smile_subsample_segments[j]
            sa = data_group[j]
            test = sa[int(smile_temp[i]): int(smile_temp[i + 1]) + 1] # current test smiles

            if data_ts is None:
                data_ts = test
            else:
                data_ts = np.vstack((test, data_ts))
            
            train = sa
            train = np.delete(train, list(range(int(smile_temp[i]), int(smile_temp[i + 1] + 1))), axis = 0)
            
            if data_tr is None:
                data_tr = train
                
            else:
                data_tr = np.vstack((train, data_tr))

        # data_ts = np.array(data_ts)
        # data_tr = np.array(data_tr)

        # mdl = KNeighborsClassifier(n_neighbors=4)

        # data_tr = (data_tr - data_tr.mean()) / data_tr.std() # To standardize the train data?

        train_data = data_tr[:, inp]


        mdl.fit(train_data, data_tr[:, -1])

        test_preds = mdl.predict(data_ts[:, inp])
        fit_temp[:, i] = np.sum([test_preds[pred] != data_ts[:, -1][pred] for pred in range(len(test_preds))]) / (len(test_preds))
        
    res = np.mean(fit_temp, axis=1)
    return res