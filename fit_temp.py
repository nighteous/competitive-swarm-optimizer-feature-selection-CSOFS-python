import numpy as np

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
        data_ts = []
        data_tr = []

        for j in range(len(classes)):
            smile_temp = smile_subsample_segments[j]
            sa = data_group[j]
            test = sa[int(smile_temp[i]): int(smile_temp[i + 1])] # current test smiles

            data_ts = [test, data_ts]
            
            train = sa
            train = np.delete(train, list(range(int(smile_temp[i]), int(smile_temp[i + 1]))), axis = 0)
            print(sa[int(smile_temp[i]): int(smile_temp[i + 1])] == train[int(smile_temp[i]): int(smile_temp[i + 1])])
            