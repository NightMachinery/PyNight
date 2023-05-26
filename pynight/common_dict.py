from types import SimpleNamespace
import uuid

##
class ReadOnlySimpleNamespace(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hash = uuid.uuid4()

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
