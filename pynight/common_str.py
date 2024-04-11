import sys


##
def print_to_file(*args, file=sys.stdout, **kwargs):
    if isinstance(file, str):
        file_obj = open(file, "a")
    else:
        file_obj = file

    try:
        print(*args, file=file_obj, **kwargs)
    finally:
        if isinstance(file, str):
            file_obj.close()


##
def try_float(value):
    try:
        return float(value)
    except ValueError:
        return value


##
def whitespace_shared_rm(input_text):
    """
    Remove the smallest shared indentation from a given string of text.

    This function converts leading tabs to 4 spaces per tab and then
    removes the smallest amount of leading spaces that is common to all lines.

    :param input_text: A string containing the text to process.
    :return: A string with leading tabs converted to spaces and the smallest
             shared leading space indentation removed from each line.

    Examples:

    >>> input_text = "\\t\\tExample with tabs\\n\\t    and spaces\\n\\n\\t\\tand a mix of both"
    >>> print(whitespace_shared_rm(input_text))
    Example with tabs
    and spaces

    and a mix of both
    """

    def replace_leading_tabs(line):
        # Replace leading tabs with four spaces
        new_line = []
        start = None
        for i, char in enumerate(line):
            if char == "\t":
                new_line.append(" " * 4)
            else:
                start = i
                break

        if start is not None:
            new_line.extend(line[start:])
        return "".join(new_line)

    # Split the input text into lines
    lines = input_text.splitlines(keepends=True)
    #: Return a list of the lines in the string, breaking at line boundaries.  Line breaks are not included in the resulting list unless keepends is given and true.

    # Convert leading tabs to 4 spaces
    lines = [replace_leading_tabs(line) for line in lines]

    # Determine the smallest shared indentation
    min_indent = None
    for line in lines:
        if line.strip():  # Skip empty lines
            indent = len(line) - len(line.lstrip(" "))
            if min_indent is None or indent < min_indent:
                min_indent = indent

    # Remove minimum indentation from each line
    if min_indent is not None:
        lines = [
            line[min_indent:] if line.startswith(" " * min_indent) else line
            for line in lines
        ]

    # Return the modified lines as a single string
    return "".join(lines)


##
