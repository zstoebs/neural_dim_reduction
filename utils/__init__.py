import os
import numpy as np
import pandas as pd

### data setup
def get_dataset():
    path = os.getcwd() # put this file in the same dir as the root dir of the training data
    data_root = os.path.join(path,'tactile-coding')
    subj_paths = [d for d in os.listdir(data_root) if int(d) <= 12] # only first 12 subjs have EEG data

    dataset = []
    for i, subj in enumerate(subj_paths):
        subj_path = os.path.join(data_root,subj,'tables')
        
        # already sorted based on source id --> rows correspond in each
        waveforms = pd.read_csv(os.path.join(subj_path,'waveforms.csv'))
        units = pd.read_csv(os.path.join(subj_path,'units.csv'))
        cell_types = units['cellType']
        
        # alignment
        waveforms['sourceId'] = units['sourceId']
        waveforms.set_index('sourceId',inplace = True)
        waveforms.columns = waveforms.columns.astype('float')
        
        waveforms['subj'] = i
        data = pd.concat([waveforms,cell_types],axis=1)
        dataset += [data]

    dataset = pd.concat(dataset)
    return dataset

# feature scaling to standard normal
def standardize(arr):
    return (arr - np.mean(arr,axis=1).reshape(-1,1)) / np.std(arr,axis=1).reshape(-1,1)

# normalize to [0,1]
def normalize(arr):
    return (arr - arr.min(axis=1).reshape(-1,1)) / (arr.max(axis=1).reshape(-1,1) - arr.min(axis=1).reshape(-1,1))
    
