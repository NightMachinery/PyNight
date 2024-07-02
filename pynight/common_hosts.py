import os


def mmd1_p():
    #: checks if `/home/mmd` exists:
    if os.path.exists("/home/mmd"):
        return True
    else:
        return False
