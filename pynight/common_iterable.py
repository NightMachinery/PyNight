import re
from typing import Iterable
import operator as op
from pynight.common_icecream import ic
import itertools


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
def list_dup_rm(lst, keep_first_p=True):
    """
    Remove duplicates from the list while preserving order.

    Parameters:
    - lst: List from which to remove duplicates.
    - keep_first_p: If True, keep the first occurrence of a duplicate; otherwise, keep the last occurrence.

    Returns:
    - List with duplicates removed.
    """
    if keep_first_p:
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)

        return result
    else:
        return list_dup_rm(lst[::-1], keep_first_p=True)[::-1]


##
def flatten1_iterable(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


##
def list_of_dict_to_dict_of_list(
    lst,
    default=None,
    warn_unused_keys=False,
    exclude_key_patterns=None,
    include_key_patterns=None,
):
    """
    Converts a list of dictionaries to a dictionary of lists.

    Parameters:
    - lst: A list of dictionaries.
    - default: The default value to use when a key is missing in a dictionary. Default is None.
    - warn_unused_keys: If True, will print a warning for keys present in any dictionary but not in the first dictionary.
    - exclude_key_patterns: A list of regex patterns. Keys matching any pattern in this list will be excluded.
    - include_key_patterns: A list of regex patterns. Only keys matching a pattern in this list will be included.

    Returns:
    A dictionary of lists. Each key from the first dictionary in lst will be a key in the result, pointing to a list of values.

    Example:
    >>> lst = [
    ...     {"a": 1, "b": 2, "c": 3},
    ...     {"a": 4, "b": 5},
    ...     {"a": 6, "c": 7, "d": 8}
    ... ]
    >>> list_of_dict_to_dict_of_list(lst, default=0, warn_unused_keys=True)
    {'a': [1, 4, 6], 'b': [2, 5, 0], 'c': [3, 0, 7]}
    """

    if not lst:
        return {}

    # Use a list to maintain the order of the keys from lst[0]
    base_keys = list(lst[0].keys())
    base_keys_set = set(base_keys)
    base_keys = lst_include_exclude(
        base_keys,
        exclude_key_patterns=exclude_key_patterns,
        include_key_patterns=include_key_patterns,
    )

    result = {key: [default] * len(lst) for key in base_keys}

    unused_keys = set() if warn_unused_keys else None

    for idx, dct in enumerate(lst):
        for key in base_keys:
            result[key][idx] = dct.get(key, default)

        if warn_unused_keys:
            current_keys = set(dct.keys())
            unused = current_keys - base_keys_set
            unused_keys.update(unused)

    if warn_unused_keys and unused_keys:
        print(f"Warning: Unused keys - {', '.join(unused_keys)}", file=sys.stderr)

    return result


##
def lst_include_exclude(
    lst,
    *,
    exclude_key_patterns=None,
    include_key_patterns=None,
):
    """
    Filters a list of strings based on the provided regex patterns.

    Parameters:
    - lst: A list of strings.
    - exclude_key_patterns: A list of regex patterns. Strings matching any pattern in this list will be excluded.
    - include_key_patterns: A list of regex patterns. Only strings matching a pattern in this list will be included.

    Returns:
    A list of filtered strings.

    Example:
    >>> lst = ["apple", "banana", "cherry", "berry", "date"]
    >>> lst_include_exclude(lst, exclude_key_patterns=["^b.*"], include_key_patterns=[".*rr.*"])
    ['cherry']
    """

    def matches_any_pattern(key, patterns):
        return any(re.search(pattern, key) for pattern in patterns)

    filtered_lst = [
        key
        for key in lst
        if (
            (not include_key_patterns or matches_any_pattern(key, include_key_patterns))
            and (
                not exclude_key_patterns
                or not matches_any_pattern(key, exclude_key_patterns)
            )
        )
    ]

    return filtered_lst


##
class HiddenList(list):
    """
    This class is merely used to signal to =torch_shape_get= etc. that its content should not be shown (e.g., to avoid clutter).
    """

    #: @seeAlso [[https://github.com/google/jax/issues/18049][Is there a way to register a particular Python object as a PyTree leaf? · Issue #18049 · google/jax]]
    pass


##
def range_contiguous_p(key: range) -> bool:
    if isinstance(key, range):
        return key.step in (1, -1)

    return False


def range_to_slice(key: range) -> slice:
    if isinstance(key, range):
        return slice(key.start, key.stop, key.step)

    return None


##
