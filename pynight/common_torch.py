import jax

import torch
import torch.nn as nn
import torchvision

from contextlib import nullcontext

import matplotlib.pyplot as plt
import gc
from .common_jupyter import jupyter_gc
from .common_numpy import hash_array_np

##
def torch_shape_get(input, size_p=False, type_only_p=False):
    total_size = 0

    def h_shape_get(x):
        nonlocal total_size

        res = ()
        if hasattr(x, "dtype"):
            res += (x.dtype,)
            if hasattr(x, "shape"):
                res += (x.shape,)

        elif hasattr(x, "shape"):
            res += (type(x), x.shape)

        if size_p and hasattr(x, "element_size") and hasattr(x, "nelement"):
            size = torch_memory_tensor(x, s=2)
            total_size += size

            res += (f"{size:.2f}MB",)

        if len(res) == 0:
            if type_only_p:
                res = type(x)
            else:
                res = x

        return res

    res = jax.tree_map(h_shape_get, input)

    if size_p:
        # return (f"total_size: {total_size:.2f}MB", res)
        return dict(total_size_mb=total_size, tree=res)
    else:
        return res


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
    plt.show()


##
def torch_gpu_memory_stats():
    #: @seeAlso `print(torch.cuda.memory_summary())`
    ###
    gigabyte = 1024**3

    allocated = torch.cuda.memory_allocated() / (gigabyte)
    reserved = torch.cuda.memory_reserved() / (gigabyte)
    print(f"gpu allocated: {allocated}\ngpu reserved: {reserved}")


def torch_memory_tensor(tensor, s=3):
    #: s=3: gigabyte
    ##
    size_in_bytes = tensor.element_size() * tensor.nelement()
    size = size_in_bytes / (1024**s)
    return size


def torch_gpu_empty_cache():
    jupyter_gc()
    gc.collect()
    torch.cuda.empty_cache()


def torch_gpu_remove_all():
    #: This does not remove all GPU tensors. I don't know why. I think it's because `del obj` is not actually deleting the tensors.
    #: [[id:68188d07-4317-412f-ab74-bd3158e2a378][How do I forcefully release the memory of a tensor? - PyTorch Forums]]
    #:
    #: * [[https://docs.python.org/3/library/gc.html][gc — Garbage Collector interface — Python 3.11.3 documentation]]
    #: ** =gc.get_objects=
    #: Returns a list of all objects tracked by the collector, excluding the list returned. If generation is not None, return only the objects tracked by the collector that are in that generation.
    ##
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                if obj.is_cuda:
                    del obj
        except:
            #: There are errors when trying to determine if these objects are tensors or not.
            pass

    torch_gpu_empty_cache()


##
def no_grad_maybe(no_grad_p):
    #: @seeAlso `torch.set_grad_enabled(mode=False)`
    ##
    if no_grad_p:
        return torch.no_grad()
    else:
        return nullcontext()


##
def model_device_get(model):
    return next(model.parameters()).device


##
def hash_tensor(tensor, *args, **kwargs):
    return hash_array_np(tensor.cpu().numpy(), *args, **kwargs)


##
def prepend_value(tensor: torch.Tensor, value) -> torch.Tensor:
    """
    Returns a tensor which is the input tensor with `value` prepended to its last dimension.

    Args:
        tensor (torch.Tensor): The input tensor.
        value: The value to prepend.

    Returns:
        torch.Tensor: The output tensor `value` prepended to its last dimension.

    Example:
        >>> tensor = torch.tensor([[2, 2, 3], [4, 5, 6]])
        >>> prepend_value(tensor, 33)
        tensor([[33,  2,  2,  3],
                [33,  4,  5,  6]])
    """
    device = tensor.device

    filler = torch.full(
        (*tensor.shape[:-1], 1), fill_value=value, dtype=tensor.dtype, device=device
    )
    return torch.cat((filler, tensor), dim=-1)


##
def drop_tokens(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    num_prefix_tokens: int = 1,
    ordered: bool = True,
) -> torch.Tensor:
    """
    Drops tokens according to the given mask and preserves the prefix tokens.

    Parameters:
    tokens: [batch, token, hidden]
        The tokens to drop.
    mask: [batch, token]
        The mask indicating which tokens to keep.
    num_prefix_tokens: int, default = 1
        The number of prefix tokens to keep.
    ordered: bool, default = True
        Whether to keep the order of the tokens or not.

    Returns:
    torch.Tensor
        The tensor with dropped tokens.

    Example:
    ```
    import torch
    from icecream import ic

    tokens = torch.randint(low=0, high=100, size=(2, 5, 10))

    mask = torch.tensor([[True, False, False, True], [False, True, True, False]], dtype=torch.int64)
    ic(mask.shape)

    dropped_tokens = drop_tokens(tokens, mask, num_prefix_tokens=1, ordered=True)
    ic(dropped_tokens.shape)

    print(tokens)
    print(mask)
    print(dropped_tokens)
    ```
    This would output:
    ```
    ic| mask.shape: torch.Size([2, 4])
    ic| dropped_tokens.shape: torch.Size([2, 3, 10])
    tensor([[[16, 65, 37, 10, 71, 13, 59, 62, 78, 91],
             [72, 76, 57, 34, 76, 91, 61,  2, 66, 28],
             [22, 92,  8, 14, 93, 12, 45, 54, 27, 38],
             [13, 37,  3, 56, 85, 24,  9, 22, 14, 97],
             [69,  3, 49, 42, 80, 82,  8, 74, 21, 67]],

            [[24, 33, 32, 39,  6, 89, 96, 90, 65, 79],
             [80,  8, 86, 93, 34, 86,  9, 75, 63, 78],
             [61, 89, 57, 55,  2, 84, 59, 65, 95, 23],
             [31, 49, 97, 28, 23, 69, 96, 58, 46, 59],
             [59, 15, 33, 29, 90, 59, 35, 16, 56, 73]]])
    tensor([[1, 0, 0, 1],
            [0, 1, 1, 0]])
    tensor([[[16, 65, 37, 10, 71, 13, 59, 62, 78, 91],
             [72, 76, 57, 34, 76, 91, 61,  2, 66, 28],
             [69,  3, 49, 42, 80, 82,  8, 74, 21, 67]],

            [[24, 33, 32, 39,  6, 89, 96, 90, 65, 79],
             [61, 89, 57, 55,  2, 84, 59, 65, 95, 23],
             [31, 49, 97, 28, 23, 69, 96, 58, 46, 59]]])
    ```
    """

    if num_prefix_tokens:
        prefix_tokens, tokens = (
            tokens[:, :num_prefix_tokens],
            tokens[:, num_prefix_tokens:],
        )
    else:
        prefix_tokens = None

    assert tokens.shape[:-1] == mask.shape

    tokens_shape = tokens.shape
    # ic(tokens_shape)

    tokens = tokens[mask.nonzero(as_tuple=True)]
    tokens = tokens.reshape((tokens_shape[0], -1, *tokens_shape[2:]))
    #: @assumption For all i, `mask[i].sum()` is constant.

    # ic(tokens.shape)

    if prefix_tokens is not None:
        # ic(prefix_tokens.shape)

        tokens = torch.cat((prefix_tokens, tokens), dim=1)

    return tokens


##
