import sys
import traceback

def traceback_print(file=None):
    print(traceback.format_exc(), file or sys.stderr)
