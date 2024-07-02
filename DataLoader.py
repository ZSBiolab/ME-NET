import numpy as np
import pandas as pd
import torch


# Diagnosis1,Diagnosis2
def sort_data(path):
    ''' sort the genomic and clinical data_old w.r.t. survival time (OS_MONTHS) in descending order
    Input:
        path: path to input dataset (which is expected to be a csv file).
    Output:
        x: sorted genomic inputs.
        yd1: Diagnosis1
        yd2: Diagnosis2
    '''
    
    data = pd.read_csv(path)

    x = data.drop(["Diagnosis1", "Diagnosis2"], axis=1).values
    yd1 = data.loc[:, ["Diagnosis1"]].values
    yd2 = data.loc[:, ["Diagnosis2"]].values
    # age = data_old.loc[:, ["AGE"]].values

    return (x, yd1, yd2)


def load_data(path, dtype, device,
              task='Diagnosis1'):
    '''Load the sorted data_old, and then covert it to a Pytorch tensor.
    Input:
        path: path to input dataset (which is expected to be a csv file).
        dtype: define the data_old type of tensor (i.e. dtype=torch.FloatTensor)
        task: 'Diagnosis1','Diagnosis2' or 'both'
    Output:
        X: a Pytorch tensor of 'x' from sort_data().

        yd1: Diagnosis1

        yd2: Diagnosis2
    '''
    print("Loading data...")
    print(path)
    data = pd.read_csv(path)
    try:
        x = data.drop(["Diagnosis1", "Diagnosis2", 'Unnamed: 0', 'Unnamed: 0.1'],
                      axis=1).values
    except KeyError:
        x = data.drop(["Diagnosis1", "Diagnosis2", 'Unnamed: 0'],
                      axis=1).values
    yd1 = data.loc[:, ["Diagnosis1"]].values
    yd2 = data.loc[:, ["Diagnosis2"]].values

    X = torch.from_numpy(x).type(dtype).to(device=device)
    yd1 = torch.from_numpy(yd1).to(device=device)
    yd2 = torch.from_numpy(yd2).to(device=device)
    print("Finish Loading")
    return (X, yd1, yd2)


def load_pathway(path, dtype, device):
    '''Load a bi-adjacency matrix of pathways, and then covert it to a Pytorch tensor.
    Input:
        path: path to input dataset (which is expected to be a csv file).
        dtype: define the data_old type of tensor (i.e. dtype=torch.FloatTensor)
    Output:
        PATHWAY_MASK: a Pytorch tensor of the bi-adjacency matrix of pathways.
    '''
    pathway_mask = pd.read_csv(path, index_col=0).values

    PATHWAY_MASK = torch.from_numpy(pathway_mask).type(dtype)
    PATHWAY_MASK = PATHWAY_MASK.to(device)

    return (PATHWAY_MASK)
