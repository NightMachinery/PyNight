##
def iterable_chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


##
def get_or_none(lst, n):
    try:
        return lst[n]
    except:
        return None


##
def grep(lst, regex):
    return list(filter(lambda x: re.search(regex, x), lst))


rg = grep


def dir_grep(obj, regex):
    return grep(dir(obj), regex)


dg = dir_grep
##
def list_mv(lst, item, final_index=0):
    lst.insert(final_index, lst.pop(lst.index(item)))
    return lst


##
def params_cartesian_gen(parameters):
    if not parameters:
        yield dict()
    else:
        key_to_iterate = list(parameters.keys())[0]
        next_round_parameters = {
            p: parameters[p] for p in parameters if p != key_to_iterate
        }
        for val in parameters[key_to_iterate]:
            for pars in params_cartesian_gen(next_round_parameters):
                temp_res = pars
                temp_res[key_to_iterate] = val
                yield temp_res


##
