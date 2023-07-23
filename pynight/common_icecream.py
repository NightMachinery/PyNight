from pynight.common_condition import jupyter_p
from icecream import ic, colorize as ic_colorize


if jupyter_p:
    ic.configureOutput(outputFunction=lambda s: print(s))
else:
    ic.configureOutput(outputFunction=lambda s: print(ic_colorize(s)))
