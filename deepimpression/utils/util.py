from torch import nn
import yaml
import os
import pickle
from typing import Tuple
import importlib


def serialize_class(cls: type) -> str:
    # Serializes a class into a string which can be used to recreate the class
    return f"{cls.__qualname__}"


def deserialize_class(class_qualname: str) -> type:
    # Deserialize a class based on its qualname
    module_name, class_name = class_qualname.rsplit('.', 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls


def save_weights(model: nn.Module, filename: str) -> None:
    # Saves weights to the given filename
    with open(filename, 'wb') as f:
        if isinstance(model, nn.DataParallel):
            pickle.dump(model.module.state_dict(), f)
        else:
            pickle.dump(model.state_dict(), f)


def save_model(model: nn.Module, params: dict, dirname: str, weights_name: str="weights.pth") -> None:
    if not os.path.exists(dirname):
        raise ValueError(f"Directory {dirname} does not exist.")
    save_weights(model, f"{dirname}/{weights_name}")

    if 'class' not in params['model']:
        params['model']['class'] = serialize_class(model.__class__)

    with open(f"{dirname}/params.yaml", 'w') as f:
        yaml.dump(params, f)


def load_weights(model: nn.Module, filename: str) -> None:
    # Loads weights to the given filename
    with open(filename, 'rb') as f:
        model.load_state_dict(pickle.load(f))


def load_params(filename: str) -> dict:
    with open(filename, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params


def load_model(dirname: str, weights_name: str="weights.pth") -> Tuple[nn.Module, dict]:
    if not os.path.exists(dirname):
        raise ValueError(f"Directory {dirname} does not exist.")

    params = load_params(f"{dirname}/params.yaml")
    cls = deserialize_class(params['model']['class'])
    model_kwargs = params['model']
    del model_kwargs['class']
    model = cls(**model_kwargs)
    load_weights(model, f"{dirname}/{weights_name}")
    return model, params
