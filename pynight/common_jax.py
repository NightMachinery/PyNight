import jax
import jax.numpy as jnp
import jax.random as jrnd

import numpy as np
from brish import z
import os
import sys
import pickle

from pynight.common_dict import (
    SimpleObject,
    rosn_split,
    rosn_tie,
    # BatchedDict,
    # batched_dict_tree_flatten,
    # batched_dict_tree_unflatten,
)

from pynight.common_iterable import IndexableList

##
def indexablelist_flatten(indexable_list):
    # The constituent data of the IndexableList is just its elements.
    # The auxiliary data is the length of the list.
    return (indexable_list, len(indexable_list))


def indexablelist_unflatten(length, elems):
    return IndexableList(elems)


jax.tree_util.register_pytree_node(
    IndexableList, indexablelist_flatten, indexablelist_unflatten
)
##
jax.tree_util.register_pytree_node(SimpleObject, rosn_split, rosn_tie)


# jax.tree_util.register_pytree_node(BatchedDict, batched_dict_tree_flatten, batched_dict_tree_unflatten)
# @broken:
# :     102 def batched_dict_tree_unflatten(aux_data, children):
# :     103     keys, values = aux_data, children
# : --> 104     return BatchedDict(zip(keys, values))
# :
# : TypeError: unhashable type: 'list'
##
def tree_save(out_dir: str, state, flat_array=False) -> None:
    z("mkdir -p {out_dir}").assert_zero
    with open(os.path.join(out_dir, "arrays.npy"), "wb") as f:
        if flat_array:
            np.save(f, state, allow_pickle=False)
        else:
            for x in jax.tree_leaves(state):
                np.save(f, x, allow_pickle=False)

    if not flat_array:
        tree_struct = jax.tree_map(lambda t: 0, state)
        with open(os.path.join(out_dir, "tree.pkl"), "wb") as f:
            pickle.dump(tree_struct, f)


def tree_restore(out_dir, flat_array=False):
    if not flat_array:
        with open(os.path.join(out_dir, "tree.pkl"), "rb") as f:
            tree_struct = pickle.load(f)
            leaves, treedef = jax.tree_flatten(tree_struct)

    with open(os.path.join(out_dir, "arrays.npy"), "rb") as f:
        if flat_array:
            return np.load(f)
        else:
            flat_state = [np.load(f) for _ in leaves]
            return jax.tree_unflatten(treedef, flat_state)


##
