from types import SimpleNamespace
import uuid


##
class ReadOnlySimpleNamespace(SimpleNamespace):
    def __init__(self, _hash=None, _drop_nones=False, **kwargs):
        if _drop_nones:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
        super().__init__(**kwargs)
        self._hash = _hash or uuid.uuid4()

    def __getitem__(self, name):
        return getattr(self, name)

    def __setattr__(self, name, value):
        if name == "_hash":
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                f"Cannot modify attribute '{name}', this namespace is read-only."
            )

    def __hash__(self):
        return hash(self._hash)

    @property
    def __dict__(self):
        return {k: v for k, v in super().__dict__.items() if k != "_hash"}

    def __contains__(self, item):
        return item in self.__dict__

def rosn_split(rosn):
    #: [[https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.register_pytree_node.html][jax.tree_util.register_pytree_node — JAX documentation]]
    return (tuple(rosn.__dict__.values()), tuple(rosn.__dict__.keys()))


def rosn_tie(keys, values):
    return ReadOnlySimpleNamespace(**dict(zip(keys, values)))


# simple_obj = SimpleNamespace
simple_obj = ReadOnlySimpleNamespace


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
