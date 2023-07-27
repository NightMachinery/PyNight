from pynight.common_condition import jupyter_p
from icecream import ic, colorize as ic_colorize


def ic_colorize2(input):
    input_str = str(input)

    return ic_colorize(input_str)
    # return input_str


if jupyter_p():
    ic.configureOutput(outputFunction=lambda s: print(s))
else:
    ic.configureOutput(outputFunction=lambda s: print(ic_colorize2(s)))
