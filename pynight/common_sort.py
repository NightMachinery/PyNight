import re
import os

##
def identity(x):
    return x


def version_sort_key(s, pre_key_fn=identity):
    """Sort strings containing numbers in a way that the numbers are considered as a whole,
    not individual characters (e.g. 10 comes after 2)"""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("(\d+)", pre_key_fn(s))
    ]


##
