import json
import os
from pynight.common_files import mkdir
from pynight.common_debugging import fn_name_current


##
class JSONEncoderWithFallback(json.JSONEncoder):
    """
    A custom JSON encoder that falls back to a user-defined function for encoding
    unsupported objects. If no custom function is provided, it defaults to using the
    built-in `str` function.

    Parameters
    ----------
    fallback_function : callable, optional
        A function that takes an unsupported object as its argument and returns a
        JSON-serializable representation of the object. If not provided, the default
        `str` function will be used. (default is `str`)
    *args : tuple
        Variable-length argument list passed to the parent class constructor.
    **kwargs : dict
        Arbitrary keyword arguments passed to the parent class constructor.

    Example
    -------
    encoder = JSONEncoderWithFallback()
    encoded = encoder.encode({"key": "value"})

    custom_fallback = lambda obj: f"Fallback: {obj}"
    custom_encoder = JSONEncoderWithFallback(fallback_function=custom_fallback, indent=2)
    encoded_custom = custom_encoder.encode({"key": "value"})
    """

    def __init__(self, *args, fallback_function=str, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_function = fallback_function

    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return self.fallback_function(obj)


def dumps(
    obj,
    indent=2,
    **kwargs,
):
    encoder = JSONEncoderWithFallback(
        indent=indent,
        **kwargs,
    )
    return encoder.encode(obj)


def json_save(
    obj,
    *,
    file,
    indent=2,
    exists_mode="ignore",
    **kwargs,
):
    json_data = dumps(obj, indent=indent, **kwargs)

    if isinstance(file, str):
        #: If file is a path string, ensure the directory exists
        mkdir(file, do_dirname=True)

        #: Check if the file exists and handle according to exists_mode
        if os.path.exists(file):
            if exists_mode == "error":
                raise FileExistsError(f"The file '{file}' already exists.")
            elif exists_mode == "ignore":
                #: Do nothing, proceed to write the file
                pass
            else:
                raise ValueError(f"Invalid exists_mode: '{exists_mode}'")

        with open(file, "w", encoding="utf-8") as f:
            f.write(json_data)
    else:
        #: If file is a file-like object, just write to it
        file.write(json_data)


def json_save_update(
    obj,
    *,
    file,
    no_overwrite_p=False,
    **kwargs,
):
    kwargs["exists_mode"] = "ignore"

    # Check if the file exists
    if isinstance(file, str) and os.path.exists(file):
        # Load the existing JSON data
        with open(file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)

        # Update the existing JSON data with the new object
        if no_overwrite_p:
            # Only add new keys, do not overwrite existing keys
            for key, value in obj.items():
                if key not in existing_data:
                    existing_data[key] = value
                else:
                    raise KeyError(
                        f"{fn_name_current()}: Key '{key}' already exists in the JSON data. (Turn off 'no_overwrite_p' to overwrite the data.)"
                    )
        else:
            # Update existing keys and add new ones

            existing_data.update(obj)

        # Use json_save to save the updated JSON data
        json_save(existing_data, file=file, **kwargs)
    else:
        # If the file does not exist or 'file' is a file-like object, just save the new object

        json_save(obj, file=file, **kwargs)


##
def json_load(path):
    with open(path, "r") as f:
        return json.load(f)


def json_partitioned_load(paths):
    output = {}
    for path in paths:
        try:
            current = json_load(path)
            output.update(current)
        except FileNotFoundError:
            print(f"File not found: {path}")
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from file: {path}")

    return output


##
