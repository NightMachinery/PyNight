from pynight.common_condition import jupyter_p
from pynight.common_debugging import traceback_print
from icecream import ic
from icecream import colorize as ic_colorize
import sys


ic_output = sys.stdout


def ic_colorize2(input):
    input_str = str(input)

    return ic_colorize(input_str)
    # return input_str


if jupyter_p():

    def _ic_print(s):
        try:
            print(
                s,
                file=ic_output,
                flush=True,
            )
        except:
            traceback_print()

else:

    def _ic_print(s):
        try:
            print(
                ic_colorize2(s),
                file=ic_output,
                flush=True,
            )
        except:
            traceback_print()


ic.configureOutput(outputFunction=_ic_print)
