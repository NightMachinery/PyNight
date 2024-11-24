import os
from os import getenv


##
def str_falsey_to_none(string):
    if not string:
        return None
    else:
        return string


##
def getenv2(injected_value, env_var, default):
    """
    Retrieves the value from either an injected value (if not None), an environment variable, or the default. Also runs str_falsey_to_none on the default value or the one from the environment.
    """

    if injected_value is None:
        return str_falsey_to_none(getenv(env_var, default=default))

    return injected_value


##
def delenv(varname: str) -> None:
    """
    Safely delete an environment variable if it exists.

    Args:
        varname (str): Name of environment variable to delete
    """
    try:
        del os.environ[varname]
    except KeyError:
        pass


##
