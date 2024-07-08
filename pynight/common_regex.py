import re


##
float_pattern = r"\d+(?:\.\d*)?"
float_pattern_compiled = re.compile(float_pattern)


##
def regex_quote(s):
    return re.escape(s)


##
def re_maybe(pattern_str):
    return f"(?:{pattern_str})?"


##
def rget(input, pattern):
    m = re.compile(f".*?{pattern}.*").match(input)
    if m:
        groups = m.groups()
        if len(groups) >= 1:
            return groups[0]

    return None


def rget_percent(text, pattern):
    number_str = rget(text, pattern)

    if number_str:
        number_float = float(number_str)

        result = number_float / 100

        return result
    else:
        return None


##
