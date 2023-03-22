import io
import pathlib

Path = pathlib.Path
##
def cat(file_path):
    with io.open(file_path, "r", encoding="utf8") as file_:
        text = file_.read()
        return text


##
def mkdir(path):
    #: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
    return Path(path).mkdir(parents=True, exist_ok=True)


dir_ensure = mkdir
ensure_dir = mkdir
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
