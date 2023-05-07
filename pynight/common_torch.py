import jax

import torch
import torch.nn as nn
import torchvision

from contextlib import nullcontext

import matplotlib.pyplot as plt
import gc
from .common_jupyter import jupyter_gc


##
def torch_shape_get(input):
    def h_shape_get(x):
        if hasattr(x, 'dtype'):
            return x.dtype, x.shape
        elif hasattr(x, 'shape'):
            return type(x), x.shape
        else:
            return x

    return jax.tree_map(h_shape_get, input)


##
class TorchModelMode:
    """
    A context manager to temporarily set a PyTorch model's mode to either training or evaluation, and restore its
    original mode when exiting the context.

    Args:
        model (torch.nn.Module): The PyTorch model whose mode needs to be temporarily set.
        mode (str): The mode to set the model to within the context. Must be either 'train' or 'eval'.

    Example:
        model = nn.Linear(10, 5)  # A simple PyTorch model for demonstration

        with TorchModelMode(model, 'eval'):
            # The model is in evaluation mode within this block
            pass

        # The model is back to its original mode after the block
    """

    def __init__(self, model: nn.Module, mode: str):
        self.model = model
        self.mode = mode
        self.original_mode = None

    def __enter__(self):
        self.original_mode = self.model.training  # Save the original mode
        if self.mode.lower() == "train":
            self.model.train()
        elif self.mode.lower() == "eval":
            self.model.eval()
        else:
            raise ValueError(f"Invalid mode '{self.mode}', must be 'train' or 'eval'")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_mode:  # Restore the original mode
            self.model.train()
        else:
            self.model.eval()


##
torch_to_PIL = torchvision.transforms.ToPILImage()

def img_tensor_show(img_tensor):
    plt.imshow(torch_to_PIL(img_tensor))
##
def torch_gpu_memory_stats():
    #: @seeAlso `print(torch.cuda.memory_summary())`
    ###
    gigabyte = 1024**3

    allocated = torch.cuda.memory_allocated()/(gigabyte)
    reserved = torch.cuda.memory_reserved()/(gigabyte)
    print(f"gpu allocated: {allocated}\ngpu reserved: {reserved}")


def torch_memory_tensor(tensor):
    size_in_bytes = tensor.element_size() * tensor.nelement()
    size_in_gigabytes = size_in_bytes / (1024 ** 3)
    return size_in_gigabytes


def torch_gpu_empty_cache():
    jupyter_gc()
    gc.collect()
    torch.cuda.empty_cache()


def torch_gpu_remove_all():
    #: This does not remove all GPU tensors. I don't know why.
    #:
    #: * [[https://docs.python.org/3/library/gc.html][gc — Garbage Collector interface — Python 3.11.3 documentation]]
    #: ** =gc.get_objects=
    #: Returns a list of all objects tracked by the collector, excluding the list returned. If generation is not None, return only the objects tracked by the collector that are in that generation.
    ##
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    del obj
        except:
            #: There are errors when trying to determine if these objects are tensors or not.
            pass

    torch_gpu_empty_cache()
##
def no_grad_maybe(no_grad_p):
    if no_grad_p:
        return torch.no_grad()
    else:
        return nullcontext()
##
