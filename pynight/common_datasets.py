from typing import Iterable
import os
from functools import wraps
from pynight.common_debugging import traceback_print
from pynight.common_iterable import (
    BatchedIterable,
    IndexableList,
    range_to_slice,
)
from pynight.common_benchmark import (
    timed,
    Timed,
)
from pynight.common_torch import torch_shape_get
from pynight.common_dict import (
    SimpleObject,
    simple_obj,
    BatchedDict,
)
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

from pynight.common_dynamic import (
    DynamicVariables,
    DynamicObject,
    dynamic_set,
    dynamic_get,
)

dynamic_vars = dict()
dynamic_obj = DynamicObject(dynamic_vars, default_to_none_p=True)
##
class TransformResult(SimpleObject):
    """
    This class is interpreted specially by `TransformedDataset`; the transform's result will be `TransformResult.result`.
    """
    pass
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
def transform_result_postprocess(data):
    if isinstance(data, TransformResult):
        data = data.result

    if isinstance(data, dict):
        data = BatchedDict(data)
    elif isinstance(data, list):
        data = IndexableList(data)

    return data


@dataclass()
class TransformedDataset:
    dataset: datasets.Dataset #: or dict
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

    def select(
        self,
        indices: Iterable,
        *args,
        **kwargs,
    ):
        if hasattr(self.dataset, "select"):
            #: [[https://github.com/huggingface/datasets/blob/2.14.5/src/datasets/arrow_dataset.py#L3731][datasets/src/datasets/arrow_dataset.py at 2.14.5 · huggingface/datasets]]
            dataset_new = self.dataset.select(indices, *args, **kwargs)
        else:
            try:
                indices = range_to_slice(indices) or indices

                dataset_new = self.dataset[indices]
            except:
                ic(
                    torch_shape_get(self.dataset, type_only_p=True,),
                    torch_shape_get(indices, type_only_p=True,)
                )

                raise

        return TransformedDataset(dataset_new, self.transforms.copy())

    def transform_columns(self, mapping, drop_unselected_p=False):
        def h_transform_columns(batch):
            new_batch = dict()
            for k, v in batch.items():
                if k in mapping:
                    new_batch[mapping[k]] = v
                elif not drop_unselected_p:
                    new_batch[k] = v

            return new_batch

        return self.transform(h_transform_columns)


    def __getitem__(self, *args, **kwargs):
        time_p = dynamic_obj.transformed_dataset_time_p
        # ic(time_p)

        with Timed(name="dataset.__getitem__", enabled_p=time_p):
            data = self.dataset.__getitem__(*args, **kwargs)

        data = transform_result_postprocess(data)
        with Timed(name="All Transforms", enabled_p=time_p):
            for transform in self.transforms:
                if time_p:
                    transform = timed(transform)

                data = transform(data)
                data = transform_result_postprocess(data)

        return data

    def __len__(self):
        return len(self.dataset)

    def batched_iterator(self, batch_size, drop_last_batch=False):
        iterable = BatchedIterable(self, batch_size, drop_last_batch)
        return iterable

    def fn_with_transforms(self, fn, time_p=False):
        @wraps(fn)
        def fn2(batch, *args, **kwargs):
            batch_transformed = transform_result_postprocess(batch)

            with Timed(name="All Transforms", enabled_p=time_p):
                for transform in self.transforms:
                    if time_p:
                        transform = timed(transform)

                    try:
                        batch_transformed = transform(batch_transformed)
                        batch_transformed = transform_result_postprocess(batch_transformed)
                    except:
                        ic(
                            torch_shape_get(batch_transformed, type_only_p=True),
                            transform,
                        )
                        raise

            return fn(*args, batch=batch, batch_transformed=batch_transformed, **kwargs)

        return fn2


@dataclass()
class ConcatenatedTransformedDataset:
    datasets: List[TransformedDataset]

    def __init__(self, datasets):
        self.datasets = datasets

    def datasets_concatenated(
        self,
    ):
        length = len(self)
        datasets_ = [tds.dataset.select(range(length)) for tds in self.datasets]

        for i in range(1, len(datasets_)):
            #: @hack HF does not allow duplicate columns
            ##
            ds = datasets_[i]

            cols_rm = [c for c in datasets_[0].column_names if c in ds.column_names]
            # ic(cols_rm)

            datasets_[i] = ds.remove_columns(cols_rm)

        ds = datasets.concatenate_datasets(datasets_, axis=1)
        return ds

    def preview(self, batch_size=2, type_only_p=True, **kwargs):
        #: @duplicateCode
        ##
        return torch_shape_get(self[:batch_size], type_only_p=type_only_p)

    def call_recursive(self, name, *args, **kwargs):
        datasets_new = [getattr(ds, name)(*args, **kwargs) for ds in self.datasets]
        return ConcatenatedTransformedDataset(datasets=datasets_new)

    def transform(self, *args, **kwargs):
        return self.call_recursive("transform", *args, **kwargs)

    def select(self, *args, **kwargs):
        return self.call_recursive("select", *args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        batch_transformed = dict()
        for ds in self.datasets:
            batch_transformed.update(ds.__getitem__(*args, **kwargs))

        return batch_transformed

    def __len__(self):
        return min(len(ds) for ds in self.datasets)

    def batched_iterator(self, batch_size, drop_last_batch=False):
        iterable = BatchedIterable(self, batch_size, drop_last_batch)
        return iterable

    def fn_with_transforms(self, fn):
        @wraps(fn)
        def fn2(batch, *args, **kwargs):
            batch_transformed = BatchedDict()
            for ds in self.datasets:
                batch_current = BatchedDict()
                for k, v in batch.items():
                    if k in ds.dataset.column_names:
                        batch_current[k] = v

                for transform in ds.transforms:
                    # ic(torch_shape_get(batch_current, type_only_p=True), transform)

                    batch_current = transform(batch_current)
                    batch_current = transform_result_postprocess(batch_current)

                batch_transformed.update(batch_current)

            return fn(*args, batch=batch, batch_transformed=batch_transformed, **kwargs)

        return fn2


##
def mapconcat(
    dataset,
    function,
    unchanged_columns=None,
    unchanged_keep_columns=True,
    time_p=False,
    **kwargs,
):
    if unchanged_columns is None:
        unchanged_columns = dataset.column_names
        if isinstance(unchanged_columns, dict):
            #: Aggregate all the lists in this dictionary to a single list:
            unchanged_columns = [
                item for sublist in unchanged_columns.values() for item in sublist
            ]
    # ic(unchanged_columns)

    @wraps(function)
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

    if time_p:
        function_wrapped = timed(function_wrapped)

    with Timed(name="dataset.map", enabled_p=time_p):
        ds_new = dataset.map(
            function_wrapped, remove_columns=unchanged_columns, **kwargs
        )

    if unchanged_keep_columns:
        with Timed(name="unchanged_keep_columns", enabled_p=time_p):
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
