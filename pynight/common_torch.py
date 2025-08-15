import sys
import numpy
import numpy as np
import torch
import torch.nn as nn
import torchvision
import functools
from contextlib import nullcontext
import socket
import psutil
import humanize
import time
import matplotlib.pyplot as plt
import gc
from math import prod
from PIL import Image
from .common_jupyter import jupyter_gc
from .common_numpy import hash_array_np
from .common_obj import delattr_force
from pynight.common_hash import is_hashable
from pynight.common_files import rm
from pynight.common_icecream import ic
from pynight.common_dict import simple_obj
from pynight.common_iterable import (
    HiddenList,
)
import torchvision.transforms as transforms


# import pynight.common_dict

try:
    import jax
except ImportError:
    pass

try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    pass


try:
    import nvidia_smi

    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False

##
torch_shape_get_hidden = set()
torch_shape_get_hidden_ids = set()


def torch_shape_get(input, size_p=False, type_only_p=False, device_p=True):
    total_size = 0

    def is_leaf(x):
        #: [[https://stackoverflow.com/questions/77269443/how-to-add-custom-attributes-to-a-python-list][How to add custom attributes to a Python list? - Stack Overflow]]
        ##
        if (
            (is_hashable(x) and x in torch_shape_get_hidden)
            or id(x) in torch_shape_get_hidden_ids
            or (hasattr(x, "_shape_get_hidden_p") and x._shape_get_hidden_p)
        ):
            return True
        else:
            return False

    def h_shape_get(x):
        if isinstance(x, dict):
            #: handles classes inheriting from dict
            #: a normal dict should never reach us, as it should be handled by =tree_map= itself
            return torch_shape_get(dict(x))

        nonlocal total_size

        res = ()
        if hasattr(x, "dtype"):
            res += (x.dtype,)
            if hasattr(x, "shape"):
                res += (x.shape,)

        elif hasattr(x, "shape"):
            res += (type(x), x.shape)

        if device_p and hasattr(x, "device"):
            res += (x.device,)

        if size_p and hasattr(x, "element_size") and hasattr(x, "nelement"):
            size = torch_memory_tensor(x, s=2)
            total_size += size

            res += (f"{size:.2f}MB",)

        if len(res) == 0:
            if type_only_p or isinstance(x, HiddenList) or is_leaf(x):
                res = type(x)
            else:
                res = x

        return res

    res = jax.tree_map(h_shape_get, input, is_leaf=is_leaf)

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


def img_tensor_show(
    img_tensor,
    dpi=100,
    title=None,
):
    if isinstance(img_tensor, Image.Image):
        image = img_tensor
    else:
        ##
        #: Check if the tensor is boolean, and if so, convert to uint8
        if img_tensor.dtype == torch.bool:
            img_tensor = img_tensor.to(torch.uint8) * 255
            #: Convert True/False to 255/0

        #: Alternatively, for NumPy arrays:
        if img_tensor.dtype == np.bool_:
            img_tensor = img_tensor.astype(np.uint8) * 255
        ##
        image = torch_to_PIL(img_tensor)

    # Get image dimensions
    height, width = image.size

    #: Set the figure size based on the image dimensions
    plt.figure(figsize=(width / dpi, height / dpi))

    plt.imshow(image)
    plt.axis("off")

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()
    plt.close()


##
def torch_gpu_memory_stats():
    #: @seeAlso `print(torch.cuda.memory_summary())`
    ###
    gigabyte = 1024**3

    allocated = torch.cuda.memory_allocated() / (gigabyte)
    reserved = torch.cuda.memory_reserved() / (gigabyte)
    print(f"gpu allocated: {allocated}\ngpu reserved: {reserved}")


def gpu_memory_get(unit="GB", mode="total"):
    """
    Get GPU memory information. Uses PyTorch for total memory and nvidia-smi for free memory.

    Args:
        unit (str): Memory unit ('GB' for gigabytes)
        mode (str): Memory mode ('total' for total memory, 'free' for free memory across all processes)

    Returns:
        float: Memory amount in specified unit
    """
    if unit.lower() == "gb":
        unit_divide_by = 1024**3

    if not torch.cuda.is_available():
        msg = "CUDA is not available. Please check your installation."
        print(msg, file=sys.stderr)
        return 0

    device = torch.cuda.current_device()

    if mode.lower() == "total":
        # Use PyTorch for total memory
        memory = torch.cuda.get_device_properties(device).total_memory
    elif mode.lower() == "free":
        # Use nvidia-smi for free memory (accounts for all processes)
        if not NVIDIA_SMI_AVAILABLE:
            msg = "nvidia-smi Python bindings not found. Install with: pip install nvidia-ml-py"
            print(msg, file=sys.stderr)
            return 0

        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        memory = info.free
        nvidia_smi.nvmlShutdown()
    else:
        raise ValueError("mode must be either 'total' or 'free'")

    memory_in_unit = memory / unit_divide_by
    return memory_in_unit


def torch_memory_tensor(tensor, s=3):
    #: s=3: gigabyte
    ##
    size_in_bytes = tensor.element_size() * tensor.nelement()
    size = size_in_bytes / (1024**s)
    return size


def torch_gpu_empty_cache(gc_mode="full"):
    if gc_mode == "full":
        jupyter_gc()
        #: does `gc.collect()`, too
    elif gc_mode == "py":
        gc.collect()
    elif gc_mode is None or gc_mode is False:
        pass
    else:
        raise ValueError(f"Invalid gc_mode: {gc_mode}")

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

    dropped_tokens = drop_tokens(tokens, mask, num_prefix_tokens=1,)
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


def mask_to_zero(x, *, keep_indices):
    """
    Zero out all tokens in the tensor `x` except those specified in `keep_indices`.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_patches, channels).
        keep_indices (torch.Tensor): Tensor of indices to keep, shape (batch_size, num_tokens_to_keep).

    Returns:
        torch.Tensor: Tensor with unspecified tokens zeroed out.
    """
    #: @unused, @untested
    batch_size, num_patches, channels = x.shape
    num_tokens_to_keep = keep_indices.shape[1]

    assert (
        keep_indices.min() >= 0 and keep_indices.max() < num_patches
    ), f"Indices out of bounds: min={keep_indices.min()}, max={keep_indices.max()}, num_patches={num_patches}"

    # Create a mask of zeros
    mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=x.device)

    # Create batch indices
    batch_indices = (
        torch.arange(batch_size, device=x.device)
        .unsqueeze(1)
        .expand(-1, num_tokens_to_keep)
    )
    #: [[https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html][torch.Tensor.expand — PyTorch 2.4 documentation]]
    #: Returns a new view of the self tensor with singleton dimensions expanded to a larger size.  Passing -1 as the size for a dimension means not changing the size of that dimension.  Tensor can be also expanded to a larger number of dimensions, and the new ones will be appended at the front. For the new dimensions, the size cannot be set to -1.

    # Set mask to True at the keep_indices
    mask[batch_indices, keep_indices] = True

    # Unsqueeze mask to match x's dimensions for broadcasting
    mask = mask.unsqueeze(-1)

    # Zero out the unspecified tokens
    x = x * mask

    return x


##
def torch_device_name_get(device=None):
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(device=device)
    else:
        device_name = "cpu"

    return device_name


##
def host_info_get(device=None):
    #: @deprecated
    #: @alt ${NIGHTDIR}/python/system_info.py
    ##
    metadata = dict()

    device_name = torch_device_name_get(device=device)

    hostname = socket.gethostname()

    total_ram = psutil.virtual_memory().total
    total_ram_gb = total_ram // (1024**3)
    #: converts bytes to gigabytes

    metadata["device_name"] = device_name
    metadata["total_ram_gb"] = total_ram_gb
    metadata["hostname"] = hostname

    try:
        device_properties = torch.cuda.get_device_properties(device)
        device_properties_dict = {
            "name": device_properties.name,
            "major": device_properties.major,
            "minor": device_properties.minor,
            "total_memory": device_properties.total_memory,
            "total_memory_humanized": humanize.naturalsize(
                device_properties.total_memory,
                binary=True,
            ),
            "multi_processor_count": device_properties.multi_processor_count,
        }
        metadata["device_properties"] = device_properties_dict
    except:
        traceback_print()
        print("Continuing despite the error ...", file=sys.stderr)

    return metadata


##
class TorchBenchmarker:
    def __init__(
        self,
        *,
        output_dict,
        device=None,
        measure_carbon_p=False,
        country_iso_code="USA",
        tracking_mode="machine",
        output_dir=None,
        output_file="emissions.csv",
        output_append_p=False,
    ):
        self.device = device
        self.metadata = output_dict

        self.measure_carbon_p = measure_carbon_p
        self.country_iso_code = country_iso_code
        self.tracking_mode = tracking_mode
        self.output_dir = output_dir
        self.output_file = output_file
        self.output_append_p = output_append_p

        # self.tracker = None
        # self.start_time = None
        # self.end_time = None
        # self.max_memory_allocated = None

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats(device=self.device)

        if self.measure_carbon_p:
            self.tracker = OfflineEmissionsTracker(
                country_iso_code=self.country_iso_code,
                tracking_mode=self.tracking_mode,
                save_to_file=bool(self.output_dir),
                output_dir=self.output_dir,
                output_file=self.output_file,
            )
            self.tracker.start()

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        device = self.device

        self.end_time = time.time()
        time_taken = self.end_time - self.start_time
        time_taken_humanized = humanize.precisedelta(time_taken)
        self.metadata["time_total"] = time_taken
        self.metadata["time_total_humanized"] = time_taken_humanized

        self.max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
        max_memory_allocated_humanized = humanize.naturalsize(
            self.max_memory_allocated,
            binary=True,
        )
        self.metadata["max_memory_allocated"] = self.max_memory_allocated
        self.metadata["max_memory_allocated_humanized"] = max_memory_allocated_humanized
        self.metadata["memory_stats"] = torch.cuda.memory_stats_as_nested_dict(
            device=device
        )

        if self.measure_carbon_p:
            if not self.output_append_p:
                rm(f"{self.output_dir}/{self.output_file}")

            self.tracker.stop()

            ##: @notNeeded
            # h(tracker.flush)
            # tracker.flush()
            ##


##
def tensorify_scalars(argnums=(0,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Store the original types of the arguments specified by argnums
            original_types = {
                argnum: torch.is_tensor(args[argnum]) for argnum in argnums
            }

            # Convert those arguments to tensors if they're not already
            new_args = list(args)
            for argnum in argnums:
                if not original_types[argnum]:
                    new_args[argnum] = torch.tensor(args[argnum])

            # Call the original function
            result = func(*new_args, **kwargs)

            # If the original type of an argument was not a tensor, convert the result back to the same type
            for argnum in original_types.keys():
                if not original_types[argnum]:
                    return result.item()

            return result

        return wrapper

    return decorator


##
def expand_as(
    src,
    shape,
    dim=-1,
):
    if dim < 0:
        dim += len(shape)

    unsqueeze_dims_before = [1] * dim
    unsqueeze_dims_after = [1] * (len(shape) - len(src.shape) - dim)

    src = src.view(*unsqueeze_dims_before, *src.shape, *unsqueeze_dims_after)
    src = src.expand(shape)
    return src


def rank_tensor(
    data,
    descending=True,
    increment=1,
    #: The ranks start from =increment=.
    dim=-1,
    reverse_p=False,
):
    #: * @tests
    #: ** `data = torch.tensor([[4.0, 2.0, 11], [9.0, 11, 7]])`
    ##
    _, sorted_indices = torch.sort(data, dim=dim, descending=descending)

    ##
    sorted_range = (
        torch.arange(
            data.shape[dim],
            device=data.device,
        )
        + increment
    )
    if reverse_p:
        sorted_range = sorted_range.flip(dims=(-1,))

    # ic(torch_shape_get((data, sorted_indices, sorted_range)))
    sorted_range = expand_as(
        sorted_range,
        data.shape,
        dim=dim,
    )

    ranks = torch.zeros_like(data, dtype=torch.int64)
    # ic(torch_shape_get((ranks, sorted_indices, sorted_range)))
    ranks = torch.scatter(input=ranks, dim=dim, index=sorted_indices, src=sorted_range)
    ##
    #: @unbatched
    # ranks = torch.zeros_like(sorted_indices)
    # ranks[sorted_indices] = torch.arange(len(sorted_indices)) + increment
    ##

    return ranks


##
def unique_first_indices(
    A,
    dim=None,
):
    """
    This function returns the first unique indices based on the input tensor.

    Parameters
    ----------
    A : torch.Tensor
        A 1-D tensor from which to select unique indices.
    dim : int, optional
        The dimension to apply uniqueness. If None, the tensor is flattened.
        Defaults to None.

    Returns
    -------
    torch.Tensor
        A tensor of indices that index the first occurrence of each unique value
        in the input tensor, sorted in the order that the unique values appear in
        the original input tensor.

    Example
    --------
    >>> import torch

    # Define tensors
    tensor_1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    tensor_2 = torch.tensor([7.0, 8.0, 9.0, 10.0, 11.0])

    # Define associated IDs
    id_1 = torch.tensor([0, 1, 2, 3, 4, 5])
    id_2 = torch.tensor([3, 4, 5, 6, 7])

    # Concatenate tensors and IDs
    all_tensors = torch.cat((tensor_1, tensor_2))
    all_ids = torch.cat((id_1, id_2))

    # Print original tensors and IDs
    print("Original Tensors:", all_tensors)
    print("Original IDs:", all_ids)

    first_indices = unique_first_indices(all_ids, dim=0)

    unique_tensors = all_tensors[first_indices]
    unique_ids = all_ids[first_indices]

    print("Unique Tensors:", unique_tensors)
    print("Unique IDs:", unique_ids)

    Output:
    Original Tensors: tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])
    Original IDs: tensor([0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7])
    Unique Tensors: tensor([ 1.,  2.,  3.,  4.,  5.,  6., 10., 11.])
    Unique IDs: tensor([0, 1, 2, 3, 4, 5, 6, 7])
    """

    unique, idx, counts = torch.unique(
        A,
        dim=dim,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]

    return first_indices


##
def seed_set(seed, cuda_deterministic=False):
    import torch
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        if cuda_deterministic:
            #: performance hit for avoiding negligible race conditions
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        torch.cuda.manual_seed_all(seed)


##
def drop_mask(
    tensor,
    mask,
    dim,
    check_mask_p=False,
):
    """Drop values in a tensor based on a mask along a specified dimension."""
    if dim < 0:
        dim += tensor.ndim

    # Use the mask to select values
    dropped = torch.masked_select(tensor, mask)

    new_shape = list(tensor.shape)
    # ic(new_shape, dim)
    if check_mask_p:
        # Compute the number of elements retained after masking for each slice along the specified dimension
        elements_retained_per_slice = mask.sum(dim=dim)

        unique_counts = torch.unique(elements_retained_per_slice)
        if len(unique_counts) > 1:
            raise ValueError("Inconsistent number of elements retained across batches.")

        new_shape[dim] = unique_counts.item()
    else:
        o = (prod(new_shape[:dim])) * (prod(new_shape[dim + 1 :]))
        new_shape[dim] = dropped.shape[0] // o

    return dropped.view(*new_shape)


def drop_topk(
    tensor,
    k,
    dim=-1,
    largest=True,
    keep_p=False,
):
    #: If dim is None, flatten the tensor and work on the last dimension
    if dim is None:
        tensor = tensor.view(-1)
        dim = -1

    #: Get the top k indices
    topk_values, indices = torch.topk(tensor, k, dim=dim, largest=largest)
    # ic(tensor.shape, indices.shape)

    #: Create a mask of the same shape as tensor
    mask = torch.ones_like(tensor, dtype=torch.bool)

    #: Set the mask values corresponding to top k indices to False
    mask.scatter_(dim, indices, 0)
    #: [[https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html][torch.Tensor.scatter_ — PyTorch 2.0 documentation]]
    #: Tensor.scatter_(dim, index, src, reduce=None)
    #: Writes all values from the tensor src into self at the indices specified in the index tensor. For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.

    if keep_p:
        mask = torch.logical_not(mask)

    ##: Drop the masked elements:
    # dropped = torch.masked_select(tensor, mask)
    # new_shape = list(tensor.shape)
    # new_shape[dim] -= k
    # dropped = dropped.view(*new_shape)
    ##
    dropped = drop_mask(
        tensor=tensor,
        mask=mask,
        dim=dim,
        check_mask_p=False,
    )
    ##
    return simple_obj(
        tensor=dropped,
        indices=indices,
        values=topk_values,
    )


def keep_topk(
    tensor,
    k,
    dim=-1,
    largest=True,
):
    return drop_topk(
        tensor,
        k=k,
        dim=dim,
        largest=largest,
        keep_p=True,
    )
    ##
    # return drop_topk(
    #     tensor,
    #     k=(tensor.shape[dim] - k),
    #     dim=dim,
    #     largest=(not largest),
    # )
    ##


def drop_from_dim(
    tensor,
    indices,
    dim,
    keep_p=False,
    ordered_p=False,
):
    mask = torch.ones_like(tensor, dtype=torch.bool)

    if ordered_p:
        assert keep_p, "When dropping, it's meaningless to preserve the order."

        #: we want to keep the values at the specified indices
        #: [[https://pytorch.org/docs/stable/generated/torch.gather.html][torch.gather — PyTorch 2.0 documentation]]
        return tensor.gather(dim, indices)

    mask.scatter_(dim, indices, 0)
    if keep_p:
        mask = torch.logical_not(mask)

    return drop_mask(
        tensor=tensor,
        mask=mask,
        dim=dim,
        check_mask_p=False,
    )


##
def scale_patch_to_pixel(
    patch_wise,
    verbose=False,
    output_channel_dim_p=False,
    output_width=None,
    output_height=None,
    interpolate_mode="nearest-exact",
    #: Mode =mode='nearest-exact'= matches Scikit-Image and PIL nearest neighbours interpolation algorithms and fixes known issues with =mode='nearest'=. This mode is introduced to keep backward compatibility. Mode =mode='nearest'= matches buggy OpenCV's =INTER_NEAREST= interpolation algorithm.
    # interpolate_mode="nearest",
):
    #: patch_wise: ([batch,] patch)
    #: patch_wise (can be pixel-wise): (batch, [1,] width, height)
    #: output: (batch, [1,] width, height)
    #: assumes square image
    ##
    output_width = output_width or output_height
    output_height = output_height or output_width
    assert output_width == output_height

    # verbose = True #: for debugging

    if verbose:
        ic(patch_wise.shape)

    if patch_wise.ndim == 3 and patch_wise.shape[-1] == patch_wise.shape[-2]:
        pixel_wise = patch_wise

        if output_channel_dim_p:
            pixel_wise = pixel_wise.unsqueeze(1)
            #: (batch, 1, width, height)

    elif patch_wise.ndim == 4 and patch_wise.shape[-1] == patch_wise.shape[-2]:
        #: (batch, 1, width, height)
        pixel_wise = patch_wise

        if not output_channel_dim_p:
            pixel_wise = pixel_wise.squeeze(1)
            #: (batch, width, height)

    else:
        patch_w = int(patch_wise.shape[-1] ** 0.5)
        patch_h = patch_w
        if verbose:
            ic(patch_w)

        #: The following reshape works both for unbatched and batched inputs, and always results in a batched output (with the batch size possibly being one).
        pixel_wise = patch_wise.reshape((-1, patch_w, patch_h))
        if verbose:
            ic(pixel_wise.shape)

        pixel_wise = pixel_wise.unsqueeze(1)
        if verbose:
            ic(pixel_wise.shape)

        pixel_wise = nn.functional.interpolate(
            pixel_wise,
            scale_factor=(output_width / patch_w),
            mode=interpolate_mode,
        )
        #: The input dimensions are interpreted in the form:
        #: `mini-batch x channels x [optional depth] x [optional height] x width`.
        if verbose:
            ic(pixel_wise.shape)

        if not output_channel_dim_p:
            pixel_wise = pixel_wise.squeeze(1)  #: remove useless channel dimension

    if verbose:
        ic(pixel_wise.shape)

    return pixel_wise


##
def swap_interpolation_to(
    transform, interpolation=transforms.InterpolationMode.NEAREST
):
    """
    Swap the interpolation mode of torchvision transforms to the specified mode.

    Args:
    - transform (transforms.Compose): The torchvision Compose transform.
    - interpolation (transforms.InterpolationMode): The desired interpolation mode.

    Returns:
    - transforms.Compose: The modified Compose transform with the updated interpolation mode.
    """
    if isinstance(transform, transforms.Compose):
        # Iterate through each transform in the Compose
        new_transforms = []
        for t in transform.transforms:
            new_transforms.append(swap_interpolation_to(t, interpolation))
        return transforms.Compose(new_transforms)
    elif hasattr(transform, "interpolation"):
        # If the transform has an interpolation attribute, create a new instance with the desired interpolation
        new_kwargs = {
            k: v
            for k, v in transform.__dict__.items()
            if k
            not in [
                "training",
                "interpolation",
            ]
            and not k.startswith("_")
        }
        return transform.__class__(interpolation=interpolation, **new_kwargs)
    else:
        # If the transform does not have an interpolation attribute, return it unchanged
        return transform


##
def quantile_sum(
    input_tensor,
    quantile,
    *,
    dim,
    keepdim=False,
    greater_than_p=True,
):
    quantile_value = torch.quantile(input_tensor, quantile, dim=dim, keepdim=True)
    #: quantile(input, q, dim=None, keepdim=False, *, interpolation='linear', out=None) -> Tensor

    if greater_than_p:
        mask = input_tensor >= quantile_value
    else:
        mask = input_tensor < quantile_value

    mask = mask.float()
    return torch.sum(input_tensor * mask, dim=dim, keepdim=keepdim)


##
def swap_backward(forward, backward):
    return forward.detach() + (backward - backward.detach())


##
def mod_copy_params_buffers(original, replacement):
    for name, param in original.named_parameters():
        if "." in name:
            print(f"skipped submodule's param: {name}")
            continue

        try:
            replacement.register_parameter(name, param)
        except:
            ic(name, torch_shape_get(param))
            raise

    for name, buffer in original.named_buffers():
        replacement.register_buffer(name, buffer)


def mod_init_from(original, replacement):
    import inspect

    kwargs = {}
    sig_keys = list(inspect.signature(original.__init__).parameters.keys())
    # sig_keys += ['prefix']

    for arg in sig_keys:
        if hasattr(original, arg):
            kwargs[arg] = getattr(original, arg)

    if hasattr(original, "bias"):
        kwargs["bias"] = True if original.bias is not None else False

    try:
        replacement = replacement(**kwargs)

        if hasattr(original, "prefix"):
            replacement.prefix = original.prefix

    except:
        ic(sig_keys, kwargs)
        raise

    mod_copy_params_buffers(original, replacement)

    #: Copy submodules
    for name, submodule in original.named_children():
        # if not hasattr(replacement, name):
        setattr(replacement, name, submodule)

    return replacement


def module_mapper(
    module,
    module_mapping,
    prefix_autoset_p=True,
):
    """
    Recursively replace modules based on the provided module_mapping.

    Args:
    module (nn.Module): The input module to be modified.
    module_mapping (dict): A dictionary mapping original module types to their replacements.

    Returns:
    SimpleObj: A named tuple containing the modified model and a set of module names not replaced.
    """
    modules_not_replaced = set()

    def _mapper(
        module,
        prefix="",
        recurse_on_mapped_p=True,
    ):
        def process_children(mod):
            not_replaced = set()
            for name, child in mod.named_children():
                child_prefix = f"{prefix}.{name}" if prefix else name
                modified_child, child_not_replaced = _mapper(
                    child, child_prefix, recurse_on_mapped_p
                )
                if modified_child is not child:
                    setattr(mod, name, modified_child)
                not_replaced.update(child_not_replaced)
            return not_replaced

        if type(module) in module_mapping:
            replacement_class = module_mapping[type(module)]
            if not isinstance(module, replacement_class):
                replacement = mod_init_from(module, replacement_class)

                if recurse_on_mapped_p:
                    process_children(replacement)

                return replacement, set()

        if len(list(module.children())) == 0:
            return module, {prefix}

        not_replaced = process_children(module)
        return module, not_replaced

    new_model, modules_not_replaced = _mapper(module)

    if prefix_autoset_p:
        new_model = torch_prefix_autoset(new_model)

    return simple_obj(
        new_model=new_model,
        modules_not_replaced=modules_not_replaced,
    )


##
def flatten_and_move_to_last(tensor, dim):
    #: @test
    # tensor = torch.randn(2, 3, 4, 5)
    # dims = [1, 2]
    # result = flatten_and_move_to_last(tensor, dims)
    # print(result.shape)  # Should print torch.Size([2, 5, 12])
    ##
    dims = dim

    shape = list(tensor.shape)
    # ic(shape, dim)

    len_shape = len(shape)

    dims_tmp = []
    for dim in dims:
        if dim < 0:
            dim += len_shape

        dims_tmp.append(dim)

    dims = dims_tmp

    #: Ensure dim is a list of unique dimensions
    dims = sorted(set(dims))

    #: Permute the dimensions to move the specified dims to the end
    permute_order = [i for i in range(len_shape) if i not in dims] + dims
    # ic(permute_order, dims)

    #: Permute the tensor
    permuted_tensor = tensor.permute(permute_order)

    #: Calculate the new shape after flattening the specified dimensions
    flattened_dim_size = 1
    for dim in dims:
        flattened_dim_size *= shape[dim]

    # Create the new shape
    new_shape = [shape[i] for i in range(len(shape)) if i not in dims]
    new_shape.append(flattened_dim_size)

    # Reshape the permuted tensor to the new shape
    reshaped_tensor = permuted_tensor.reshape(new_shape)

    # ic(shape, dims, permute_order, new_shape)

    return reshaped_tensor


##
def nan_to_0(tensor):
    tensor[torch.isnan(tensor)] = 0
    return tensor


##
def get_compact_gbrand(
    ig_steps=None,
    gradient_mode_brand="NG",
    patchifier_gbrand=None,
    linear_ds_gbrand=None,
    qkv_ds_gbrand=None,
    mlp_ds_gbrand=None,
    mlp_mul_gbrand=None,
    softmax_mode="S0",
    normalize_to_unit_vector_gbrand=None,
):
    if gradient_mode_brand in ["GlobALTI"]:
        compact_gbrand = gradient_mode_brand

    elif True:
        parts = [gradient_mode_brand or ""]

        if ig_steps:
            parts.append(f"ig{ig_steps}")

        if softmax_mode and softmax_mode not in ["S0"]:
            parts.append(f"h.{softmax_mode}")

        if patchifier_gbrand:
            parts.append(f"p.{patchifier_gbrand}")

        if linear_ds_gbrand:
            parts.append(f"lin_ds.{linear_ds_gbrand}")

        if qkv_ds_gbrand:
            parts.append(f"qkv_ds.{qkv_ds_gbrand}")

        if mlp_ds_gbrand:
            parts.append(f"mlp_ds.{mlp_ds_gbrand}")

        if mlp_mul_gbrand:
            parts.append(f"mlp_x.{mlp_mul_gbrand}")

        if normalize_to_unit_vector_gbrand:
            parts.append(f"n2u.{normalize_to_unit_vector_gbrand}")

        compact_gbrand = ",".join(parts)

    else:
        #: @deprecated No longer updated
        compact_gbrand = f"""{gradient_mode_brand or ""},h.{softmax_mode or ""},p.{patchifier_gbrand or ""},lin_ds.{linear_ds_gbrand or ""},mlp_ds.{mlp_ds_gbrand or ""},mlp_x.{mlp_mul_gbrand or ""}"""

    return compact_gbrand


##
def torch_prefix_autoset(
    root_mod,
    prefix="",
    # prefix="from_class",
):
    def get_prefix(module):
        prefix = ""
        if hasattr(module, "prefix") and not module.prefix.startswith("unset."):
            prefix = module.prefix

        return prefix

    def set_prefix(module, prefix=""):
        if prefix == "from_class":
            prefix = f"{module.__class__.__name__}."

        mod_prefix = get_prefix(module)
        if not prefix:
            prefix = mod_prefix

        if prefix and not mod_prefix:
            module.prefix = prefix

        for name, child in module.named_children():
            child_prefix = f"{prefix}{name}." if prefix else f"{name}."
            set_prefix(child, child_prefix)

    set_prefix(root_mod, prefix=prefix)

    return root_mod


##
def grey_tensor_to_pil(some_tensor, *, colormap="viridis", normalize_p=True):
    # Ensure the tensor is on CPU
    some_tensor = some_tensor.detach().cpu()

    # Squeeze the tensor to remove any singleton dimensions
    some_tensor = some_tensor.squeeze()

    # Check if the resulting tensor has 2 dimensions
    if some_tensor.dim() != 2:
        raise ValueError(
            f"Expected a 2D tensor after squeezing, but got {some_tensor.dim()}D"
        )

    # Convert to numpy array
    np_array = some_tensor.numpy()

    # Normalize if requested
    if normalize_p:
        np_array = (np_array - np_array.min()) / (np_array.max() - np_array.min())

    # Apply colormap
    colored_array = plt.get_cmap(colormap)(np_array)

    # Convert to uint8 and create PIL Image
    colored_array = (colored_array * 255).astype(np.uint8)
    image_pil = Image.fromarray(colored_array)

    return simple_obj(
        image_pil=image_pil,
    )


##
def decomposed_inputs_p(obj):
    return hasattr(obj, "decomposition_config") or hasattr(obj, "attributions_v")


##
def tensor_register_hook(
    tensor,
    *args,
    **kwargs,
):
    if tensor.requires_grad:
        tensor.register_hook(
            *args,
            **kwargs,
        )
    else:
        #: We could have a flag that will optionally raise an exception here.
        pass


def store_tensor_with_grad(
    tensor,
    *,
    store_in,
    name_out=None,
    name_grad_out=None,
    enabled_out_p=True,
    enabled_grad_out_p=True,
):
    if decomposed_inputs_p(tensor):
        print("store_tensor_with_grad: skipped for decomposed input", file=sys.stderr)
        return

    if name_out is not None:
        if enabled_out_p:
            setattr(store_in, name_out, tensor.detach().cpu())

        else:
            delattr_force(store_in, name_out)

    if name_grad_out is not None:
        delattr_force(store_in, name_grad_out)

        if enabled_grad_out_p:

            def store_value_grad(grad):
                # ic(torch_shape_get(grad))
                setattr(store_in, name_grad_out, grad.detach().cpu())

            tensor_register_hook(tensor, store_value_grad)


##
def get_k_predicted_indices(logits, *, k):
    """
    Retrieves the k-th most probable indices from the logits tensor.

    Args:
        logits (torch.Tensor): The input logits tensor of shape (batch_size, num_classes).
        k (int): The rank of the predicted index to retrieve (1 for first, 2 for second, etc.).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the k-th predicted indices.

    Raises:
        ValueError: If k is not within the valid range [1, num_classes].
                    If logits does not have shape (batch_size, num_classes).
    """
    # Validate logits tensor dimensions
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be a 2D tensor of shape (batch_size, num_classes), but got shape {logits.shape}"
        )

    num_classes = logits.size(-1)
    if num_classes == 0:
        raise ValueError("num_classes must be greater than zero")

    # Validation to ensure k is within the valid range
    if not (1 <= k <= num_classes):
        raise ValueError(
            f"k must be between 1 and the number of classes ({num_classes}), got k={k}"
        )

    # Obtain the top k indices along the last dimension
    logits_topk_indices = torch.topk(
        logits,
        k,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    # logits_topk_indices has shape (batch_size, k)

    # Select the k-th predicted index (0-based indexing)
    kth_predicted_indices = logits_topk_indices[..., k - 1]
    # kth_predicted_indices has shape (batch_size,)

    return kth_predicted_indices


def get_first_predicted_indices(logits):
    """
    Retrieves the most probable (first) indices from the logits tensor.

    Args:
        logits (torch.Tensor): The input logits tensor of shape (batch_size, num_classes).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the first predicted indices.
    """
    return get_k_predicted_indices(logits, k=1)


def get_second_predicted_indices(logits):
    """
    Retrieves the second most probable indices from the logits tensor.

    Args:
        logits (torch.Tensor): The input logits tensor of shape (batch_size, num_classes).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the second predicted indices.
    """
    return get_k_predicted_indices(logits, k=2)


##
