import hashlib
from urllib.parse import urlparse
import os


def hash_url(url, length=10):
    basename = os.path.basename(urlparse(url).path)

    hash_object = hashlib.sha1(url.encode('utf-8'))

    short_hash = hash_object.hexdigest()[:length]

    if basename:
        return f"{basename}_{short_hash}"
    else:
        return short_hash
