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

    if image_np.ndim == 2:  #: the image is probably grayscale
        if accept_gray_p:
            image_np = image_np[:, :, np.newaxis]
        else:
            return None

    assert (
        image_np.ndim == 3
    ), f"image_np has unexpected shape: {image_np.shape}, URL: {url}"
    #: (width, height, channels)

    if drop_alpha:
        image_np = image_np[:, :, :3]

    # Save to cache if cache directory is specified
    if cache_file_path is not None:
        np.save(cache_file_path, image_np)

    return image_np


##
def nanlen(arr):
    return len(arr) - np.sum(np.isnan(arr))


##
def nan_corrcoef(
    x,
    y,
    **kwargs,
):
    """
    @LLMGenerated

    Calculate the Pearson correlation coefficient between two arrays, ignoring any NaN values.

    Parameters:
    x (array_like): 1-D array containing data with potentially missing values.
    y (array_like): 1-D array containing data with potentially missing values.

    Returns:
    numpy.ndarray: A 2x2 array with the correlation coefficients.

    Raises:
    ValueError: If x and y have different lengths.

    Example:
    >>> x = np.array([1, 2, np.nan, 4, 5])
    >>> y = np.array([5, np.nan, 7, 1, 3])
    >>> nan_corrcoef(x, y)
    array([[1.        , 0.75592895],
           [0.75592895, 1.        ]])

    Note:
    The function returns a full 2x2 correlation matrix. The correlation coefficient between the non-NaN
    parts of x and y is found at indices [0, 1] or [1, 0] in the returned matrix.
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y):
        raise ValueError("Arrays x and y must be the same length.")

    # Create a boolean mask for values where neither x nor y is NaN
    mask = ~np.isnan(x) & ~np.isnan(y)

    # Apply the mask to both x and y
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Calculate the correlation coefficient using the filtered data
    return np.corrcoef(x_filtered, y_filtered, **kwargs)


##
