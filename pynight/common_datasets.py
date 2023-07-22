import os
from pynight.common_debugging import traceback_print
from pynight.common_torch import torch_shape_get
from pynight.common_dict import simple_obj
from pynight.common_files import (
    rm,
    mkdir,
)
import datasets
from collections.abc import Mapping, MutableMapping
from datasets import concatenate_datasets
import time
from dataclasses import dataclass
from typing import List, Callable
from icecream import ic

##
def dataset_push_from_disk_to_hub(path, *args, **kwargs):
    """
    Examples:
    >>> dataset_push_from_disk_to_hub('/opt/decompv/datasets/ImageNet1K-val_indexed', "NightMachinery/ImageNet1K-val-indexed")
    """

    dataset = datasets.load_from_disk(path)

    while True:
        try:
            return dataset.push_to_hub(*args, **kwargs)
        except:
            traceback_print()

            time.sleep(1)


##
def dataset_cache_filenames(dataset, cache_only_p=False, sort_p=True, **kwargs):
    res = list(set(d["filename"] for d in dataset.cache_files))

    if cache_only_p:
        res = [p for p in res if os.path.basename(p).startswith("cache-")]

    res = list(set(res))
    #: removes any potential duplicates
    #: no sorted set in stdlib

    if sort_p:
        res.sort(**kwargs)

    return res


##
@dataclass()
class TransformedDataset:
    dataset: datasets.Dataset
    transforms: List[Callable]

    def __init__(self, dataset: datasets.Dataset, transforms: List[Callable] = None):
        self.dataset = dataset
        self.transforms = transforms if transforms is not None else []

    def preview(self, batch_size=2, type_only_p=True, **kwargs):
        return torch_shape_get(self[:batch_size], type_only_p=type_only_p)

    def transform(self, new_transform: Callable):
        new_transforms = self.transforms.copy()
        new_transforms.append(new_transform)

        return TransformedDataset(self.dataset, new_transforms)

    def select(self, *args, **kwargs):
        dataset_new = self.dataset.select(*args, **kwargs)
        return TransformedDataset(dataset_new, self.transforms.copy())

    def __getitem__(self, *args, **kwargs):
        data = self.dataset.__getitem__(*args, **kwargs)
        for transform in self.transforms:
            data = transform(data)
        return data

    def __len__(self):
        return len(self.dataset)

    def batched_iterator(self, batch_size, drop_last_batch=False):
        length = len(self)

        num_batches = length // batch_size
        if not drop_last_batch:
            num_batches += length % batch_size != 0

        for i in range(num_batches):
            yield self[i * batch_size : (i + 1) * batch_size]

    def fn_with_transforms(self, fn):
        def fn2(batch, *args, **kwargs):
            batch_transformed = dict(batch)

            for transform in self.transforms:
                batch_transformed = transform(batch_transformed)

            return fn(batch, batch_transformed, *args, **kwargs)

        return fn2


##
def mapconcat(
    dataset, function, unchanged_columns=None, unchanged_keep_columns=True, **kwargs
):
    if unchanged_columns is None:
        unchanged_columns = dataset.column_names
        if isinstance(unchanged_columns, dict):
            #: Aggregate all the lists in this dictionary to a single list:
            unchanged_columns = [
                item for sublist in unchanged_columns.values() for item in sublist
            ]
    # ic(unchanged_columns)

    def function_wrapped(*args, **kwargs):
        #: Call the original function
        original_output = function(*args, **kwargs)

        #: Make sure the output is a dictionary
        if not (isinstance(original_output, (dict, MutableMapping))):
            raise ValueError("Output of the function is not a dictionary.")

        #: Remove unchanged_columns from the dictionary
        for column in unchanged_columns:
            if column in original_output:
                del original_output[column]

        return original_output

    ds_new = dataset.map(function_wrapped, remove_columns=unchanged_columns, **kwargs)

    if unchanged_keep_columns:
        if unchanged_keep_columns is True:
            unchanged_keep_columns = unchanged_columns

        ds_source_columns = dataset.select_columns(unchanged_keep_columns)
        # ic(unchanged_columns, ds_source_columns.column_names, ds_new.column_names)

        format_source = dict(ds_source_columns.format)
        format_source["columns"] += ds_new.format["columns"]

        ds_combined = concatenate_datasets([ds_source_columns, ds_new], axis=1)
        ds_combined.set_format(**format_source)

        # ic(dataset.format, ds_source_columns.format, ds_new.format, ds_combined.format, format_source)

        return ds_combined
    else:
        return ds_new


##
def h_dataset_index_add(batch, index):
    batch["id"] = index

    return batch


def dataset_index_add(
    dataset,
    num_proc=64,
    batch_size=1000,
):
    #: * @tests
    #: ** `dataset_indexed.shuffle().sort('id')['train'][10:13]`
    ##
    # dataset_indexed = dataset.map(
    #     h_dataset_index_add, with_indices=True, batched=True, num_proc=num_proc, batch_size=batch_size
    # )

    dataset_indexed = mapconcat(
        dataset,
        h_dataset_index_add,
        with_indices=True,
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
    )

    return dataset_indexed


##
def save_and_delete(dataset, dataset_path, delete_p=True, **kwargs):
    to_delete = dataset_cache_filenames(dataset, cache_only_p=True)

    dataset = dataset.flatten_indices()
    mkdir(dataset_path)  #: This path is a dir
    dataset.save_to_disk(dataset_path=dataset_path, **kwargs)
    print(f"Saved dataset to: {dataset_path}")

    dataset = datasets.load_from_disk(dataset_path)

    if delete_p:
        for p in to_delete:
            res = rm(p)
            print(res.msg)

    return simple_obj(
        dataset=dataset,
        to_delete=to_delete,
    )


##
