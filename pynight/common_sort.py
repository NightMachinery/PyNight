from pynight.common_icecream import ic
import re
import os


##
def identity(x):
    return x


def version_sort_key(
    s,
    pre_key_fn=identity,
    float_p=False,
    sign_pattern=r"(?:-|\+)?",
):
    #: @alt [[https://github.com/SethMMorton/natsort][SethMMorton/natsort: Simple yet flexible natural sorting in Python.]]
    ##
    """Sort strings containing numbers in a way that the numbers are considered as a whole,
    not individual characters (e.g. 10 comes after 2)"""

    int_pattern = re.compile(f"({sign_pattern}\\d+)")
    float_pattern = re.compile(f"({sign_pattern}(?:\\d+\\.\\d+|\\d+))")

    split_pattern = float_pattern if float_p else int_pattern

    s_processed = str(pre_key_fn(s))
    parts = list(split_pattern.split(s_processed))
    result = []
    # ic(s, s_processed, split_pattern, parts)

    for text in parts:
        # ic(text)

        try:
            if float_p:
                num = float(text)
            else:
                num = int(text)

            result.append(num)
        except ValueError:
            # If it's not a number, treat it as a string
            result.append(text)

    # ic(result)
    return result


##
