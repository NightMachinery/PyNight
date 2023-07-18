import os
import io
import pathlib
from pathlib import Path
import shutil
from typing import Union
from pynight.common_dict import simple_obj


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
