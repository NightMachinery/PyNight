from pynight.common_icecream import ic
import re
import os


##
def identity(x):
    return x


def version_sort_key(s, pre_key_fn=identity, float_p=False):
    """Sort strings containing numbers in a way that the numbers are considered as a whole,
    not individual characters (e.g. 10 comes after 2)"""

    int_pattern = re.compile("(\d+)")
    float_pattern = re.compile("(\d+\.\d+|\d+)")

    split_pattern = float_pattern if float_p else int_pattern

    result = []
    for text in split_pattern.split(str(pre_key_fn(s))):
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
