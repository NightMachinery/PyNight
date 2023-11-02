import os
import numpy
import numpy as np
import hashlib
import matplotlib.pyplot as plt


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
def image_url2np(
    url,
    format=None,
    drop_alpha=True,
    cache_dir="/opt/decompv/cache",
    accept_gray_p=False,
    # cache_dir=None,
):
    if format is None:
        format = url.split(".")[-1]

    # Define cache file path
    cache_file_path = None

    # Check if URL is a local file path
    if os.path.exists(url):
        image_np = plt.imread(url, format=format)
    else:
        # Calculate hash of the URL
        url_hash = hashlib.sha256(url.encode()).hexdigest()

        if cache_dir is not None:
            mkdir(cache_dir)

            cache_file_path = os.path.join(cache_dir, url_hash + ".npy")

            # If file is cached, load it and return
            if os.path.exists(cache_file_path):
                return np.load(cache_file_path)

        with urllib.request.urlopen(url) as url_response:
            image_data = url_response.read()

        # Convert the image data to a numpy array
        image_np = plt.imread(io.BytesIO(image_data), format=format)

    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)

    if image_np.ndim == 2: #: the image is probably grayscale
        if accept_gray_p:
            image_np = image_np[:, :, np.newaxis]
        else:
            return None

    assert image_np.ndim == 3, f"image_np has unexpected shape: {image_np.shape}, URL: {url}"
    #: (width, height, channels)

    if drop_alpha:
        image_np = image_np[:, :, :3]

    # Save to cache if cache directory is specified
    if cache_file_path is not None:
        np.save(cache_file_path, image_np)

    return image_np


##
