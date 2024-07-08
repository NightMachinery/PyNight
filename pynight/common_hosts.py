import os


def hostname_get():
    return os.uname()[1]


##
def mmd1_p():
    #: checks if `/home/mmd` exists:
    if os.path.exists("/home/mmd"):
        return True
    else:
        return False


##
def mb2_p():
    #: hostname == 'mb2.local'
    return hostname_get() == "mb2.local"


##
