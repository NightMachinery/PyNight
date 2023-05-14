import re


##
def rget(input, pattern):
    m = re.compile(f".*?{pattern}.*").match(input)
    if m:
        groups = m.groups()
        if len(groups) >= 1:
            return groups[0]

    return None


##
