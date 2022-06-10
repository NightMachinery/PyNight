import jax
import jax.numpy as jnp
import jax.random as jrnd

import numpy as np
from brish import z
import os
import sys
import pickle

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
