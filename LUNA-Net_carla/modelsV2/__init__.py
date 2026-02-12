"""
SNE-RoadSegV2 Models Package

This package contains the V2 implementation with:
- Swin Transformer backbone
- HF²B (Heterogeneous Feature Fusion Block)
- Simplified decoder with DSConv
- Fallibility-aware loss
"""

import importlib
import torch.nn as nn


def find_model_using_name(model_name):
    """Import the module "modelsV2/[model_name]_model.py"."""
    model_filename = "modelsV2." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, nn.Module):
            model = cls
    if model is None:
        print(f"In {model_filename}.py, there should be a subclass of nn.Module with class name that matches {target_model_name} in lowercase.")
        exit(0)
    return model


def create_model(opt, dataset):
    """Create a model given the option."""
    model = find_model_using_name(opt.model)
    instance = model()
    instance.initialize(opt, dataset)
    print(f"model [{type(instance).__name__}] was created")
    return instance
