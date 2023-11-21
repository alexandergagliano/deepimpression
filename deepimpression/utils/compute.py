import torch
from torch import nn


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_parallel(model):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model)
