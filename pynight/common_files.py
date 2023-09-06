import os
import re
import io
import pathlib
from pathlib import Path
import shutil
from typing import Union
from pynight.common_dict import simple_obj
from pynight.common_icecream import ic
from pynight.common_iterable import to_iterable


##
def cat(file_path):
    with io.open(file_path, "r", encoding="utf8") as file_:
        text = file_.read()
        return text


##
def mkdir(path, do_dirname=False):
    if do_dirname:
        path = os.path.dirname(path)

    #: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
    return Path(path).mkdir(parents=True, exist_ok=True)


dir_ensure = mkdir
ensure_dir = mkdir


##
def sanitize_filename(some_str):
    for x in (
        "/",
        "\\",
        "~",
    ):
        some_str = some_str.replace(x, "_")

    return some_str


##
def rm(path):
    path = Path(path)  #: make sure path is a Path object

    if not path.exists():
        return simple_obj(
            retcode=1,
            msg=f"The path {path} does not exist.",
        )

    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)  #: remove dir and all contains
    else:
        return simple_obj(
            retcode=2,
            msg=f"Unknown type: {path}",
        )

    return simple_obj(
        retcode=0,
        msg=f"Removed: {path}",
    )


##
def list_children(
    directory,
    *,
    include_patterns=None,
    abs_include_patterns=None,
    exclude_patterns=None,
    abs_exclude_patterns=None,
    recursive=False,
    sorted=True,
    verbose_p=False,
):
    children = []
    include_patterns = to_iterable(include_patterns)
    exclude_patterns = to_iterable(exclude_patterns)
    abs_include_patterns = to_iterable(abs_include_patterns)
    abs_exclude_patterns = to_iterable(abs_exclude_patterns)

    for root, dirs, files in os.walk(directory, topdown=True):
        dirs_orig = list(dirs) #: shallow copy
        if not recursive:
            dirs[:] = []
            #: When topdown is true, the caller can modify the dirnames list in-place (e.g., via del or slice assignment), and walk will only recurse into the subdirectories whose names remain in dirnames; this can be used to prune the search, or to impose a specific order of visiting.

        for child in dirs_orig + files:
            path = os.path.join(root, child)

            if verbose_p:
                ic(child, path)

            if include_patterns:
                should_include = False
                for pat in include_patterns:
                    if re.search(pat, child):
                        should_include = True
                        break
                if not should_include:
                    continue

            if abs_include_patterns:
                should_include = False
                for pat in abs_include_patterns:
                    if re.search(pat, path):
                        should_include = True
                        break
                if not should_include:
                    continue

            if exclude_patterns:
                should_exclude = False
                for pat in exclude_patterns:
                    if re.search(pat, child):
                        should_exclude = True
                        break
                if should_exclude:
                    continue

            if abs_exclude_patterns:
                should_exclude = False
                for pat in abs_exclude_patterns:
                    if re.search(pat, path):
                        should_exclude = True
                        break
                if should_exclude:
                    continue

            children.append(path)

    if sorted:
        children.sort()

    return children

##
def sanitize_filename(some_str):
    for x in (
            '/',
            '\\',
            '~',
            ):
        some_str = some_str.replace(x, '_')

    return some_str
##
