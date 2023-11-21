from deepimpression.common.data import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_train_data_loader(params):
    return get_data_loader(params['train_input'])


def get_eval_data_loader(params):
    return get_data_loader(params['eval_input'])


def get_data_loader(params):
    ## Load data from numpy files
    X = np.load(params['x'])['arr_0']
    y = np.load(params['y'])['arr_0']

    ## Swap last two axes so channels are first
    X = np.swapaxes(X, 1, 2)

    ## binarize labels 
    y = OneHotEncoder(max_categories=3, sparse_output=False).fit_transform(y.reshape(-1, 1))

    batch_size = params['batch_size']

    train_ds = CustomDataset(X, y)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=params['shuffle'])
