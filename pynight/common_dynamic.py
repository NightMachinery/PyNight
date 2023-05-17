from contextvars import ContextVar


def dynamic_set(dynamic_dict, var_name, new_value):
    """Sets a new value for a dynamic variable in the given dynamic dictionary."""
    var_object = dynamic_dict.setdefault(var_name, ContextVar(var_name))
    return var_object.set(new_value)


def dynamic_get(dynamic_dict, var_name):
    """
    Retrieve the value of a dynamic variable in the given dynamic dictionary.

    Args:
        dynamic_dict (dict): The dictionary where dynamic variables are stored.
        var_name (str): The name of the dynamic variable to retrieve.

    Returns:
        The current value of the dynamic variable, or raises a LookupError if the variable
        is not set in the current context.
    """
    if var_name in dynamic_dict:
        return dynamic_dict[var_name].get()
    else:
        raise LookupError(
            f"Dynamic variable '{var_name}' is not set in the current context."
        )


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
    with DynamicVariables(dynamic_dict, x=10):
        print(dynamic_dict['x'].get())  # prints: 10
        with DynamicVariables(dynamic_dict, x=20):
            print(dynamic_dict['x'].get())  # prints: 20
        print(dynamic_dict['x'].get())  # prints: 10

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
        self.dynamic_dict = dynamic_dict
        self.new_values = new_values
        self.tokens_dict = {}

    def __enter__(self):
        """
        Set the new values for the dynamic variables.

        This method is automatically called when entering the 'with' statement.
        It iterates over the new_values dictionary, and for each item, it sets
        a new value for the corresponding dynamic variable in dynamic_dict.
        The tokens returned by var.set(new_value) are stored in tokens_dict for later use.
        """
        for var_name, new_value in self.new_values.items():
            self.tokens_dict[var_name] = dynamic_set(
                self.dynamic_dict, var_name, new_value
            )

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
