import mimetypes
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
def mkdir(
    path,
    do_dirname=False,
):
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
        dirs_orig = list(dirs)  #: shallow copy
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
        "/",
        "\\",
        "~",
    ):
        some_str = some_str.replace(x, "_")

    return some_str


##
class open_file:
    def __init__(
        self,
        file_path,
        mode="r",
        *,
        mkdir_p=True,
        exists=None,
        **kwargs,
    ):
        self.file_path = file_path
        self.mode = mode
        self.exists = exists
        self.file = None
        self.mkdir_p = mkdir_p
        self.kwargs = kwargs

    def __enter__(self):
        if self.exists is None or self.exists in [
            "ignore",
            "overwrite",
        ]:
            pass
        elif self.exists == "error" and os.path.exists(self.file_path):
            raise FileExistsError(f"File '{self.file_path}' already exists.")
        elif self.exists == "increment_number":
            index = 1
            file_name, file_ext = os.path.splitext(self.file_path)
            while os.path.exists(self.file_path):
                self.file_path = f"{file_name}_{index}{file_ext}"
                index += 1
        else:
            raise ValueError(f"Invalid exists_mode: '{self.exists}'")

        if self.mkdir_p:
            mkdir(self.file_path, do_dirname=True)

        self.file = open(
            self.file_path,
            self.mode,
            **self.kwargs,
        )
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()


##
def hdd_free_get(
    *,
    unit="GB",
    path="/",
):
    total, used, free = shutil.disk_usage(path)

    if unit == "GB":
        free_space = free / (1024**3)  # Convert to GB
    elif unit == "MB":
        free_space = free / (1024**2)  # Convert to MB
    elif unit == "TB":
        free_space = free / (1024**4)  # Convert to TB
    elif unit == "B":
        free_space = free
    else:
        raise ValueError(f"Invalid unit: '{unit}'")

    return free_space


##
def mime_guess(
    file_path,
    *,
    backend="magic",
):
    """
    Guesses the MIME type of a file using different backends with lazy loading.

    This function imports the required libraries only when the corresponding
    backend is selected.

    Args:
        file_path (str): The path to the file.
        backend (str, optional): The backend to use for guessing the MIME type.
            Defaults to "magic".
            Possible values are:
            - "magic": Uses the `python-magic` library. Most accurate.
            - "puremagic": A pure Python alternative to `python-magic`.
            - "filetype": A small and dependency-free pure Python library.
            - "mimetypes": Uses Python's built-in `mimetypes` module, which
              guesses based on the file extension.

    Returns:
        str or None: The guessed MIME type as a string (e.g., 'image/jpeg'),
                     or None if the MIME type cannot be determined.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    if backend == "magic":
        try:
            import magic
        except ImportError:
            raise ImportError(
                "The 'python-magic' library is not installed. "
                "Please install it using 'pip install python-magic-bin' (Windows) "
                "or 'pip install python-magic' (Linux/macOS)."
            )
        try:
            mime = magic.Magic(mime=True)
            return mime.from_file(file_path)
        except Exception:
            return None

    elif backend == "puremagic":
        try:
            import puremagic
        except ImportError:
            raise ImportError(
                "The 'puremagic' library is not installed. "
                "Please install it using 'pip install puremagic'."
            )
        try:
            results = puremagic.magic_file(file_path)
            return results[0].mime_type if results else None
        except Exception:
            return None

    elif backend == "filetype":
        try:
            import filetype
        except ImportError:
            raise ImportError(
                "The 'filetype' library is not installed. "
                "Please install it using 'pip install filetype.py'."
            )
        try:
            kind = filetype.guess(file_path)
            return kind.mime if kind else None
        except Exception:
            return None

    elif backend == "mimetypes":
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type

    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            "Available backends are: 'magic', 'puremagic', 'filetype', 'mimetypes'."
        )


##
