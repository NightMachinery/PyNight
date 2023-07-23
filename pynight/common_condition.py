import os
from os import getenv

##
def jupyter_p():
    return getenv('INSIDE_JUPYTER_NOTEBOOK_P', default='') == 'y'
##
