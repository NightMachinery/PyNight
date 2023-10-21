import os
import re
import urllib
import requests
from brish import z, zp


##
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


##
def pysocks_proxy_from_env(env_var_names=None):
    if env_var_names is None:
        env_var_names = ['HTTP_PROXY', 'HTTPS_PROXY', 'SOCKS_PROXY', 'ALL_PROXY']

    proxy = {
        'proxy_type': None,
        'addr': None,
        'port': None,
        'username': None,
        'password': None,
        'rdns': True
    }

    for env_var_name in env_var_names:
        proxy_url = os.environ.get(env_var_name)
        if proxy_url:
            # Regular expression to parse the proxy URL
            proxy_pattern = re.compile(r'(https?|socks5)://([^:@]+)(?::(\d+))?(?::([^@]+):([^@]+))?')
            match = proxy_pattern.match(proxy_url)

            if match:
                groups = match.groups()
                proxy['proxy_type'] = groups[0]
                proxy['addr'] = groups[1]
                proxy['port'] = int(groups[2]) if groups[2] else None
                proxy['username'] = groups[3] if groups[3] else None
                proxy['password'] = groups[4] if groups[4] else None
                return proxy

    return None


##
