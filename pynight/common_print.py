from pprint import pformat
from typing import Any

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer

from pynight.common_icecream import ic_colorize2
from pynight.common_clipboard import clipboard_copy


##
def print_copy(
    obj,
    **kwargs,
):
    clipboard_copy(obj)
    print(
        obj,
        **kwargs,
    )


##
def pprint_color(
    obj: Any,
    color="ic",
    end=None,
    pformat_opts=None,
    **kwargs,
) -> None:
    """Pretty-print in color."""
    if pformat_opts is None:
        pformat_opts = dict()

    res = pformat(
        obj,
        **pformat_opts,
    )

    if not color:
        pass
    elif str(color) == "ic":
        res = ic_colorize2(res)
    elif str(color) == "256":
        res = highlight(
            res,
            PythonLexer(),
            Terminal256Formatter(),
        )

        if end is None:
            end = ""
            #: =highlight= adds a newline.
    else:
        raise ValueError(f"Unknown color: {color}")

    if end is None:
        end = "\n"

    print(
        res,
        end=end,
        **kwargs,
    )


##
