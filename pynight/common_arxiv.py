import re


def normalize_path_for_arxiv(
    path,
    *,
    disallowed_regex=r"[ :&\\\"]",
):
    """
    Normalize a file path by replacing spaces, slashes, colons, ampersands,
    and backslashes with underscores.

    Args:
        path (str): The file path to normalize

    Returns:
        str: The normalized file path
    """
    path = re.sub(disallowed_regex, "_", path)

    return path
