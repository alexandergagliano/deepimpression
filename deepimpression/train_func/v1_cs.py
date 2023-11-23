import cerebras_pytorch as cstorch

import tqdm
from typing import Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from deepimpression.common.metrics import update_confusion_matrix, metrics
from deepimpression.utils.util import save_model, deserialize
from deepimpression.utils.compute import get_device, data_parallel





def train(params: dict, model: Module, train_dl: DataLoader, test_dl: DataLoader):
    train_params = params['train_config']

    # Create optimizer and loss
    loss_cls = deserialize(train_params['loss']['class'])
    loss = loss_cls()
    opt_cls = deserialize(train_params['opt']['class'])
    opt = opt_cls(model.parameters(), lr=train_params['opt']['lr'])

    @cstorch.trace
    def training_step(inputs, targets):
        outputs = model(inputs)
        ic(type(inputs), inputs.shape, inputs.dtype)
        ic(type(targets), targets.shape, targets.dtype)
        ic(type(outputs), outputs.shape, outputs.dtype)
        loss_val = loss(outputs, targets)

        loss_val.backward()
        opt.step()
        opt.zero_grad()

        return loss   

    executor = cstorch.utils.data.DataExecutor(
        train_dl, num_steps=params['train_config']['epochs']*8000, checkpoint_steps=8000)

    for inputs, targets in executor:
        loss = training_step(inputs, targets)
