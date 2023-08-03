from contextvars import ContextVar
from pynight.common_icecream import ic
import functools


def dynamic_object_p(obj):
    # return (isinstance(obj, DynamicObject))
    #: Using =isinstance= can fail when hotreloading.

    return (
        hasattr(obj, "_dynamic_dict")
        and obj.__class__.__name__ == "DynamicObject"
    )


def dynamic_set_v1(dynamic_dict, name, value):
    """Sets a new value for a dynamic variable in the given dynamic dictionary."""
    if dynamic_object_p(dynamic_dict):
        dynamic_dict = dynamic_dict._dynamic_dict

    var_object = dynamic_dict.setdefault(name, ContextVar(name))
    return var_object.set(value)


def dynamic_set(dynamic_dict, *args, **kwargs):
    """
    Set new values for multiple dynamic variables in the given dynamic dictionary.

    Args:
        dynamic_dict (dict): The dictionary where dynamic variables are stored.
        *args: Pairs of variable names and new values, specified as positional arguments.
        **kwargs: New values for the dynamic variables, specified as keyword arguments.

    Returns:
        A dictionary mapping variable names to the tokens returned by var.set(new_value).
    """
    if len(args) % 2 != 0:
        raise ValueError("dynamic_set: expected an even number of arguments in 'args'")

    if dynamic_object_p(dynamic_dict):
        dynamic_dict = dynamic_dict._dynamic_dict

    tokens = {}

    for i in range(0, len(args), 2):
        var_name, new_value = args[i], args[i + 1]
        tokens[var_name] = dynamic_set_v1(dynamic_dict, var_name, new_value)

    for var_name, new_value in kwargs.items():
        tokens[var_name] = dynamic_set_v1(dynamic_dict, var_name, new_value)

    return tokens


def dynamic_get(dynamic_dict, var_name, default="MAGIC_THROW_EXCEPTION_13369831"):
    """
    Retrieve the value of a dynamic variable in the given dynamic dictionary.

    Args:
        dynamic_dict (dict): The dictionary where dynamic variables are stored.
        var_name (str): The name of the dynamic variable to retrieve.
        default: The default value to return if the dynamic variable is not set in the current context.

    Returns:
        The current value of the dynamic variable, or the default value if the variable
        is not set in the current context, or raises a LookupError if the variable
        is not set in the current context and no default value is provided.
    """
    if dynamic_object_p(dynamic_dict):
        dynamic_dict = dynamic_dict._dynamic_dict

    if var_name in dynamic_dict:
        try:
            return dynamic_dict[var_name].get()
        except LookupError:
            if default == "MAGIC_THROW_EXCEPTION_13369831":
                raise
            else:
                return default
    else:
        if default == "MAGIC_THROW_EXCEPTION_13369831":
            raise LookupError(
                f"Dynamic variable '{var_name}' is not set in the current context."
            )
        else:
            return default


class DynamicObject:
    """
    A class for managing dynamic variables.

    Dynamic variables are stored in a dictionary and each variable is associated
    with a ContextVar instance. DynamicObject provides the ability to set and get
    dynamic variables using object attribute access.

    Example usage:

    dynamic_dict = {}
    obj = DynamicObject(dynamic_dict, default_to_none_p=True)
    obj.x = -13
    obj.z = 72
    print(f"x: {obj.x}, z: {obj.z}, nonexistent_attribute: {obj.nonexistent_attribute or 'some_default_value'}")
    #: x: -13, z: 72, nonexistent_attribute: some_default_value
    with DynamicVariables(obj, x=10):
        print(f"x: {obj.x}, z: {obj.z}") #: x: 10, z: 72
        with DynamicVariables(obj, x=20):
            print(f"x: {obj.x}, z: {obj.z}") #: x: 20, z: 72
            obj.x = 99
            obj.y = 81
            obj.z = -6
            print(f"x: {obj.x}, y: {obj.y}, z: {obj.z}")
            #: x: 99, y: 81, z: -6
        print(f"x: {obj.x}, y: {obj.y}, z: {obj.z}")
        #: x: 10, y: 81, z: -6
        #: z and y did NOT get reset as they were not explicitly set in the previous context manager.

        obj.y = 560
    print(f"x: {obj.x}, y: {obj.y}, z: {obj.z}")
    #: x: -13, y: 560, z: -6

    The above code demonstrates how DynamicVariables can be nested and how the
    dynamic variables are reset when a context exits.
    """

    def __init__(self, dynamic_dict, default_to_none_p=False):
        """
        Initialize DynamicObject with a dictionary for storing the dynamic variables.

        Args:
            dynamic_dict (dict): The dictionary for storing the dynamic variables.
            default_to_none_p (bool): If True, dynamic variables are defaulted to None when retrieved and they don't exist.
            Defaults to False.
        """
        super().__setattr__("_dynamic_dict", dynamic_dict)
        super().__setattr__("_default_to_none_p", default_to_none_p)

    def __getitem__(self, name):
        """
        Retrieve the value of a dynamic variable.

        This method is automatically called when getting an attribute using square brackets.
        It calls dynamic_get to retrieve the value of the dynamic variable.

        Args:
            name (str): The name of the dynamic variable to retrieve.

        Returns:
            The current value of the dynamic variable, or raises a LookupError
            if the variable is not set in the current context and no default value is provided.
        """
        return self.__getattr__(name)

    def __setitem__(self, name, value):
        """
        Set a new value for a dynamic variable.

        This method is automatically called when setting an attribute using square brackets.
        It calls dynamic_set to set a new value for the dynamic variable.

        Args:
            name (str): The name of the dynamic variable to set a new value for.
            value: The new value for the dynamic variable.
        """
        dynamic_set(self._dynamic_dict, name, value)

    def __getattr__(self, name):
        """
        Retrieve the value of a dynamic variable.

        This method is automatically called when getting an attribute using dot notation.
        It calls dynamic_get to retrieve the value of the dynamic variable.

        Args:
            name (str): The name of the dynamic variable to retrieve.

        Returns:
            The current value of the dynamic variable, or raises a LookupError
            if the variable is not set in the current context and no default value is provided.
        """
        #: @LLM
        #: In Python, the __getattr__ method is a special method that's only invoked when the attribute wasn't found the usual ways. It's not invoked for attributes that exist on the instance dictionary.
        #: To intercept and override access to the attributes in the instance dictionary, you can use =__getattribute__= in Python. This special method is called when an attribute is accessed on an object, regardless of whether the attribute is present in the instance dictionary or not.
        #: In contrast to =__getattr__=, =__getattribute__= is called every time an attribute is accessed, and can therefore be used to intercept and customize any attribute access. It's important to be careful when using =__getattribute__= to avoid infinite recursion, as calling =self.attr= within =__getattribute__= will call =__getattribute__= again.
        if name in ("_dynamic_dict", "_default_to_none_p", "__getattr__"):
            raise AttributeError(
                f"@impossible Getting the attribute {name} is not supposed to invoke this function!"
            )

        if self._default_to_none_p:
            return dynamic_get(self._dynamic_dict, name, default=None)
        else:
            return dynamic_get(self._dynamic_dict, name)

    def __setattr__(self, name, value):
        """
        Set a new value for a dynamic variable.

        This method is automatically called when setting an attribute using dot notation.
        It calls dynamic_set to set a new value for the dynamic variable.

        Args:
            name (str): The name of the dynamic variable to set a new value for.
            value: The new value for the dynamic variable.
        """
        if name in ("_dynamic_dict", "_default_to_none_p"):
            # super().__setattr__(name, value)
            raise AttributeError("_dynamic_dict is a private variable of DynamicObject")
        else:
            dynamic_set(self._dynamic_dict, name, value)


class DynamicVariables:
    """
    A context manager class for managing dynamic variables.

    The dynamic variables are stored in a dictionary and each variable
    is associated with a ContextVar instance. Within the context managed
    by DynamicVariables, the dynamic variables can be set to new values,
    and when the context exits, the variables are automatically reset to their
    previous values.

    Example usage:

    dynamic_dict = {}
    dynamic_set(dynamic_dict, x=-13)
    with DynamicVariables(dynamic_dict, x=10):
        print(dynamic_get(dynamic_dict, 'x'))  # prints: 10
        with DynamicVariables(dynamic_dict, x=20):
            print(dynamic_get(dynamic_dict, 'x'))  # prints: 20
        print(dynamic_get(dynamic_dict, 'x'))  # prints: 10
    print(dynamic_get(dynamic_dict, 'x'))  # prints: -13

    The above code demonstrates how DynamicVariables can be nested and how the
    dynamic variables are reset when a context exits.
    """

    def __init__(self, dynamic_dict, **new_values):
        """
        Initialize DynamicVariables with a dictionary for storing the dynamic
        variables and their new values for the context.

        Args:
            dynamic_dict (dict): The dictionary for storing the dynamic variables.
            new_values: The new values for the dynamic variables.
        """
        if dynamic_object_p(dynamic_dict):
            dynamic_dict = dynamic_dict._dynamic_dict

        assert isinstance(
            dynamic_dict, dict
        ), f"Invalid type for dynamic_dict: {type(dynamic_dict)}"

        self.dynamic_dict = dynamic_dict
        self.new_values = new_values

    def __enter__(self):
        """
        Set the new values for the dynamic variables.

        This method is automatically called when entering the 'with' statement.
        It iterates over the new_values dictionary, and for each item, it sets
        a new value for the corresponding dynamic variable in dynamic_dict.
        The tokens returned by var.set(new_value) are stored in tokens_dict for later use.
        """
        self.tokens_dict = dynamic_set(self.dynamic_dict, **self.new_values)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Reset the dynamic variables to their previous values.

        This method is automatically called when exiting the 'with' statement.
        It iterates over the tokens_dict, and for each item, it uses the token
        to reset the corresponding dynamic variable in dynamic_dict to its previous value.
        """
        for var_name, token in self.tokens_dict.items():
            var_object = self.dynamic_dict[var_name]
            var_object.reset(token)

##
def partial_dynamic(fn, *, dynamic_dict, **dynamic_mappings):
    @functools.wraps(fn)
    def fn2(*args, **kwargs):
        with DynamicVariables(
                dynamic_dict,
                **dynamic_mappings,
        ):
            return fn(*args, **kwargs)

    return fn2
##
