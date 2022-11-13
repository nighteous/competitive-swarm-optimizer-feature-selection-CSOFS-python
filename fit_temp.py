import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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
    
    for i in range(len(classes)):

        sa = []
        sa = np.array([data[ind] for ind in range(len(group)) if group[ind] == classes[i]])
        
        number_of_smile_samples, _ = sa.shape
        smile_temp = np.round(np.linspace(1, number_of_smile_samples, k + 1))

        data_group.append(sa)
        smile_subsample_segments.append(smile_temp)

    fit_temp = np.zeros((1, 10))

    for i in range(k - 1):
        data_tr = None
        data_ts = None

        for j in range(len(classes)):
            smile_temp = smile_subsample_segments[j]
            sa = data_group[j]
            test = sa[int(smile_temp[i]) - 1: int(smile_temp[i + 1])] # current test smiles

            if data_ts is None:
                data_ts = test
            else:
                data_ts = np.vstack((test, data_ts))
            
            train = sa
            train = np.delete(train, list(range(int(smile_temp[i]) - 1, int(smile_temp[i + 1]))), axis = 0)
            
            if data_tr is None:
                data_tr = train
                
            else:
                data_tr = np.vstack((train, data_tr))

        # data_ts = np.array(data_ts)
        # data_tr = np.array(data_tr)

        mdl = KNeighborsClassifier(n_neighbors=4)
        mdl.fit(data_tr[:, inp], data_tr[:, -1])

        x = mdl.predict(data_ts[:, inp])
        fit_temp[:, i] = np.sum([pred for pred in range(len(x)) if x[pred] != data_ts[:, i][pred]])
        print(fit_temp)

    return np.mean(fit_temp, axis=1)