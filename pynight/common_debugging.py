import sys
import traceback
import os

debug_p = 'DEBUGME' in os.environ

def traceback_print(file=None):
    print(traceback.format_exc(), file or sys.stderr)
