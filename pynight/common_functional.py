from .common_regex import rget


##
def fn_name(fn, module_p=True):
    ##
    # return rget(str(fn), r'^<function (\S+)')
    ##
    #: [[https://stackoverflow.com/questions/58108488/what-is-qualname-in-python][What is __qualname__ in python? - Stack Overflow]]
    if not hasattr(fn, "__qualname__"):
        if hasattr(fn, "func"):
            #: a functools.partial wrapped function
            fn = fn.func

    res = ""
    if module_p:
        res += f"{fn.__module__}."

    res += f"{fn.__qualname__}"
    return res

    # return fn.__name__
    ##


##
