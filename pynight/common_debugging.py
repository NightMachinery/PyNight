import sys
import traceback
import os

debug_p = os.environ.get("DEBUGME", None)


def traceback_print(file=None):
    print(traceback.format_exc(), file or sys.stderr)
