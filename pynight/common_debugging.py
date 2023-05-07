import sys
import traceback
import os

debug_p = os.environ.get("DEBUGME", None)


def traceback_print(file=None):
    print(traceback.format_exc(), file or sys.stderr)


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

    return formatted_stacktrace


def stacktrace_caller_line():
    return stacktrace_get(back=0, skip_last=3, mode="line")


##
