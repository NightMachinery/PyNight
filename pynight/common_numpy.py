import numpy
import numpy as np
import hashlib

##
def hash_array_np(array: np.ndarray, hash_algorithm: str = "sha256") -> str:
    """
    Hashes a numpy array with the specified hashing algorithm.

    Args:
        array (np.ndarray): The numpy array to be hashed.
        hash_algorithm (str, optional): The hashing algorithm to use. Defaults to 'sha256'.

    Returns:
        str: The resulting hash.

    Example:
        >>> arr = np.array([1, 2, 3])
        >>> hash_numpy_array(arr)
        '2ef7bde608ce5404e97d5f042f95f89f1c232871'
    """
    array_data = array.tobytes()

    hash_func = hashlib.new(hash_algorithm)
    hash_func.update(array_data)

    hash_str = hash_func.hexdigest()

    return hash_str


##
