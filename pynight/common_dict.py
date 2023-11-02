from types import SimpleNamespace
from collections.abc import Mapping
from typing import List, Union, Any, Dict
import uuid
from pynight.common_iterable import (
    IndexableList,
    list_of_dict_to_dict_of_list,
)


##
class SimpleObject(SimpleNamespace, Mapping):
    def __init__(self, _hash=None, _drop_nones=False, _readonly_p=True, **kwargs):
        if _drop_nones:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
        super().__init__(**kwargs)

        super().__setattr__("_readonly_p", _readonly_p)
        self._hash = _hash or uuid.uuid4()

    def __getitem__(self, name):
        return getattr(self, name)

    def __setattr__(self, name, value):
        if name == "_hash" or (not self._readonly_p):
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                f"Cannot modify attribute '{name}', this namespace is read-only."
            )

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __hash__(self):
        return hash(self._hash)

    @property
    def __dict__(self):
        return {
            k: v
            for k, v in super().__dict__.items()
            if k not in ("_hash", "_readonly_p")
        }

    def __iter__(self):
        #: [[https://stackoverflow.com/questions/46626955/override-dict-on-class][python - Override dict() on class - Stack Overflow]]
        ##
        yield from self.__dict__.items()

    def __len__(self):
        return len(self.__dict__)

    def __contains__(self, item):
        return item in self.__dict__

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return ((key, getattr(self, key)) for key in self.keys())

    def values(self):
        return (getattr(self, key) for key in self.keys())

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __eq__(self, other):
        if not isinstance(other, SimpleObject):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)


def rosn_split(rosn):
    #: [[https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.register_pytree_node.html][jax.tree_util.register_pytree_node — JAX documentation]]
    return (tuple(rosn.__dict__.values()), tuple(rosn.__dict__.keys()))


def rosn_tie(keys, values):
    return SimpleObject(**dict(zip(keys, values)))


# simple_obj = SimpleNamespace
simple_obj = SimpleObject


def simple_obj_update(obj, *args, **kwargs):
    assert (len(args) % 2) == 0, "key-values should be balanced!"

    d = vars(obj)
    d = dict(d)  #: copies the dict, otherwise it will mutate the obj
    d.update(kwargs)

    for i in range(0, len(args), 2):
        value = args[i + 1]
        key = args[i]
        key = key.replace(".", "__")

        d[key] = value

    updated_obj = simple_obj(**d)


##
class BatchedDict(dict):
    #: @assumes the first dimension of values is the batch dict.
    #: We can extend it to get the =batch_dim= and support tuples for the dimensions.
    ##
    def __len__(self):
        #: @assumes all the values have the same length, otherwise this operation is undefined anyway.
        for _, v in self.items():
            return len(v)

    def __getitem__(self, key):
        # Check if the key is a slice
        if isinstance(
            key,
            (
                slice,
                list,
                range,
            ),
        ):
            sliced_dict = BatchedDict()

            for k, v in self.items():
                key_current = key
                if isinstance(v, (list,)):
                    #: list indices must be integers or slices
                    v = IndexableList(v)

                sliced_value = v.__getitem__(key_current)

                sliced_dict[k] = sliced_value

            return sliced_dict
        if isinstance(key, int):
            sliced_dict = dict()

            for k, v in self.items():
                sliced_value = v.__getitem__(key)

                sliced_dict[k] = sliced_value

            return sliced_dict
        else:
            return super().__getitem__(key)


def batched_dict_tree_flatten(batched_dict):
    keys, values = zip(*batched_dict.items())
    return (keys, values)


def batched_dict_tree_unflatten(aux_data, children):
    keys, values = aux_data, children
    return BatchedDict(zip(keys, values))


##
def key_del(
    dict,
    key,
    exception_if_nonexistent_p=False,
):
    if key in dict:
        del dict[key]
        return True
    elif exception_if_nonexistent_p:
        raise ValueError(f"Key not in the given dict: {key}")
    else:
        return False


##
def list_of_dict_to_bacthed_dict(
    *args,
    **kwargs,
):
    return BatchedDict(
        list_of_dict_to_dict_of_list(
            *args,
            **kwargs,
        )
    )


##
def concatenate_batches(batches: List[Dict]) -> Dict:
    #: Write `concatenate_batches(batches: List[dict])` which concatenates the given batch dicts into a single batch dict. PyTorch tensors will be concatenated using PyTorch, while Python lists will be concatenated using `+`. Dicts will be concatenated recursively.
    ##
    import torch

    def concatenate_items(items):
        first_item = items[0]

        # If the item is a tensor, concatenate using torch.cat
        if isinstance(first_item, torch.Tensor):
            return torch.cat(items, dim=0)

        # If the item is a list, concatenate using +
        elif isinstance(first_item, list):
            return sum(items, [])

        # If the item is a dict, concatenate recursively
        elif isinstance(first_item, dict):
            return concatenate_batches(items)

        else:
            raise ValueError(f"Unsupported type for concatenation: {type(first_item)}")

    # We assume that each batch has the same keys
    # Therefore, we can take the keys from the first batch as reference
    keys = batches[0].keys()

    # Initialize a dictionary to store the concatenated batches
    concatenated_batch = {}

    for key in keys:
        # For each key, collect the items across all batches
        items_to_concatenate = [batch[key] for batch in batches]
        concatenated_batch[key] = concatenate_items(items_to_concatenate)

    return concatenated_batch


##
