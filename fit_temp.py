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
    print(classes)

