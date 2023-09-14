import re
from typing import Iterable
import operator as op
from pynight.common_icecream import ic


##
def iterable_chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


##
def get_or_none(lst, n):
    try:
        return lst[n]
    except:
        return None


##
def grep(lst, regex):
    return list(filter(lambda x: re.search(regex, x), lst))


rg = grep


def dir_grep(obj, regex):
    return grep(dir(obj), regex)


dg = dir_grep


##
def list_mv(lst, item, final_index=0):
    lst.insert(final_index, lst.pop(lst.index(item)))
    return lst


##
def params_cartesian_gen(parameters):
    if not parameters:
        yield dict()
    else:
        key_to_iterate = list(parameters.keys())[0]
        next_round_parameters = {
            p: parameters[p] for p in parameters if p != key_to_iterate
        }
        for val in parameters[key_to_iterate]:
            for pars in params_cartesian_gen(next_round_parameters):
                temp_res = pars
                temp_res[key_to_iterate] = val
                yield temp_res


##
def lst_filter_out(lst, items):
    """Removes specified items from the given list and returns a new list.

    Args:
        lst (list): The list from which items should be removed.
        items (list): The items to be removed from the lst.

    Returns:
        list: A new list consisting of elements from lst that are not in items.
    """
    return [element for element in lst if element not in items]


list_rm = lst_filter_out


##


class IndexableList(list):
    def _recursive_get(self, indices):
        """Recursively get items based on nested indices."""
        #: [[https://stackoverflow.com/questions/64181453/fastest-method-for-extracting-sub-list-from-python-list-given-array-of-indexes][Fastest method for extracting sub-list from Python list given array of indexes - Stack Overflow]]
        ##
        super_obj = super()

        if isinstance(
            indices,
            (
                # list,
                Iterable,
            ),
        ) and not isinstance(indices, str):
            ##
            # return [super_obj.__getitem__(i) for i in indices]
            ##
            getter = op.itemgetter(*indices)
            res = getter(self)
            return list(res) if len(indices) > 1 else [res]
        # elif isinstance(indices, list) and all(isinstance(i, list) for i in indices):
        #     return [self._recursive_get(i) for i in indices]
        else:
            # assert isinstance(indices, (int, slice))

            return super_obj.__getitem__(indices)

    def __getitem__(self, indices):
        try:
            import torch

            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()
        except ImportError:
            pass

        return self._recursive_get(indices)


##
def to_iterable(possibly_iterable):
    if possibly_iterable is None:
        return []
    elif isinstance(possibly_iterable, (str,)):
        return [possibly_iterable]
    elif isinstance(possibly_iterable, Iterable):
        return possibly_iterable
    else:
        return [possibly_iterable]


##
class BatchedIterable:
    def __init__(self, data, batch_size, drop_last_batch=False):
        self.data = data
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch

    def __iter__(self):
        length = len(self.data)

        num_batches = length // self.batch_size
        if not self.drop_last_batch:
            num_batches += length % self.batch_size != 0

        for i in range(num_batches):
            yield self.data[i * self.batch_size : (i + 1) * self.batch_size]

    def __len__(self):
        length = len(self.data)
        num_batches = length // self.batch_size
        if not self.drop_last_batch:
            num_batches += length % self.batch_size != 0
        return num_batches


##
