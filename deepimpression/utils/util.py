from torch import nn
import yaml
import os
import pickle
from typing import Tuple, Union, Callable, Optional
import importlib


def serialize(obj: Union[type, Callable]) -> str:
    # Serializes a class into a string which can be used to recreate the class
    return f"{obj.__qualname__}"


def deserialize(qualname: str) -> type:
    # Deserialize a class based on its qualname
    module_name, obj_name = qualname.rsplit('.', 1)
    return getattr(importlib.import_module(module_name), obj_name)


def load_v1(params: dict) -> object:
    if 'class' in params:
        kwargs = params.copy()
        cls = deserialize(kwargs['class'])
        del kwargs['class']
        return cls(**kwargs)
    elif 'func' in params:
        return deserialize(params['func'])
    else:
        raise ValueError("Params must contain either 'class' or 'func' key")


def load_v2(params: dict) -> object:
    if 'class' in params:
        cls = deserialize(params['class'])
        return cls(params)
    else:
        raise ValueError("Params must contain 'class' key")


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


def load_model(
    config: Optional[str] = None,
    modeldir: Optional[str] = None,
    weights_name: str="weights.pth") -> Tuple[nn.Module, dict]:
    params = None
    if config is not None:
        if not os.path.exists(config):
            raise ValueError(f"Config file {config} does not exist.")
        params = load_params(config)
    if not os.path.exists(modeldir):
        raise ValueError(f"Directory {modeldir} does not exist.")

    if params is None:
        params = load_params(f"{dirname}/params.yaml")
    cls = deserialize(params['model']['class'])
    kwargs = params['model'].copy()
    del kwargs['class']
    model = cls(**kwargs)
    weights_file = f"{modeldir}/{weights_name}"
    if os.path.exists(weights_file):
        load_weights(model, weights_file)
    return model, params
