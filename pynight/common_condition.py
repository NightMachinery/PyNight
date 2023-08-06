import os
from os import getenv

##
def jupyter_p():
    return jupyter_p_v1() or jupyter_p_v2()
    
def jupyter_p_v1():
    return getenv('INSIDE_JUPYTER_NOTEBOOK_P', default='') == 'y'

def jupyter_p_v2():
    #: [[https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook][python - How can I check if code is executed in the IPython notebook? - Stack Overflow]]
    ##
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
##
