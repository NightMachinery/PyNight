import os
import urllib
import requests
from brish import z, zp


def proxy_set(proxy_address=None):
    if proxy_address:
        os.environ["ALL_PROXY"] = proxy_address
        os.environ["all_proxy"] = proxy_address
        os.environ["http_proxy"] = proxy_address
        os.environ["https_proxy"] = proxy_address
        os.environ["HTTP_PROXY"] = proxy_address
        os.environ["HTTPS_PROXY"] = proxy_address

        proxies = {"http": proxy_address, "https": proxy_address}
        proxy_handler = urllib.request.ProxyHandler(proxies)
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)
    else:
        #: Remove environmental variables related to proxies
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)

        #: Remove proxy settings from urllib.request
        proxies = dict()
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)

    return os.environ.get("HTTP_PROXY", None)
