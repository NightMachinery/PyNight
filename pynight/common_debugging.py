import inspect
import re
import sys
import traceback
import os
import importlib
from types import ModuleType
from pynight.common_condition import jupyter_p


debug_p = os.environ.get("DEBUGME", None)
deus_p = os.environ.get("deusvult", None)


def traceback_print(file=None):
    print(traceback.format_exc(), file or sys.stderr)


##
def ipdb_enable(
    disable_in_jupyter_p=True,
):
    if disable_in_jupyter_p:
        if jupyter_p():
            return

    import sys
    from IPython.core import ultratb

    sys.excepthook = ultratb.FormattedTB(
        mode="Verbose",
        color_scheme="Linux",
        call_pdb=1,
    )


##
def fn_name_current(back=1):
    frame = inspect.currentframe()
    for _ in range(back):
        frame = frame.f_back
    return frame.f_code.co_name


##
def stacktrace_get(back=0, skip_last=2, mode="line"):
    """
    Returns a formatted stacktrace string, showing a specified number of calls
    before the current function call. This is useful for debugging purposes.

    Args:
        back (int, optional): Number of calls to show before the current function call. Defaults to 1.
        skip_last (int, optional): Number of calls to skip at the end of the stacktrace. Defaults to 1.
        mode (str, optional): Determines the level of detail in the stacktrace. If 'full', it includes
            file name, line number, and function name. If 'line', it only includes the line of code.
            Defaults to 'full'.

    Returns:
        str: Formatted stacktrace string.
    """

    back += 1  #: to account for this function

    stack = traceback.extract_stack()

    end_index = -1 - skip_last
    start_index = end_index - (back)

    formatted_stacktrace = f""
    for frame in stack[start_index:end_index]:
        if mode == "full":
            formatted_stacktrace += f'File "{frame.filename}", line {frame.lineno}, in {frame.name}\n    {frame.line}\n'
        elif mode == "line":
            formatted_stacktrace += f"{frame.line}\n"

    return formatted_stacktrace.rstrip()


def stacktrace_caller_line():
    return stacktrace_get(back=0, skip_last=3, mode="line")


##
def reload_modules(target):
    """
    Reloads the specified package and all its submodules in the sys.modules dictionary.

    This function iterates through the sys.modules dictionary and reloads every module that
    starts with the specified package's name or the provided string or matches the regex pattern.
    It returns a list of names of the reloaded modules.

    Args:
        target (module, str, or re.Pattern): The package or module that should be reloaded
            along with its submodules. If provided as a string or regex pattern, it should
            match the desired package or module names.

    Returns:
        reloaded_names (list): A list of names of the reloaded modules.

    Examples:
        >>> reload_modules(pynight)

        >>> import pynight.common_debugging as common_debugging
        >>> reload_modules(common_debugging)

        >>> reload_modules(re.compile("^pynight"))

        >>> reload_modules("pynight") # equivalent to the above example
    """
    reloaded_names = []

    if isinstance(target, ModuleType):
        target_name = target.__name__
        target_pattern = None
    elif isinstance(target, str):
        target_name = target
        target_pattern = None
    elif isinstance(target, re.Pattern):
        target_name = None
        target_pattern = target
    else:
        raise TypeError(
            "Invalid target type. Must be a module, string, or regex pattern."
        )

    for name, module in list(sys.modules.items()):
        do_reload = False
        if isinstance(module, ModuleType):
            if target_name and name.startswith(target_name):
                reloaded_names.append(name)
                do_reload = True
            elif target_pattern and target_pattern.match(name):
                reloaded_names.append(name)
                do_reload = True

            if do_reload:
                importlib.reload(module)

    return reloaded_names


##
