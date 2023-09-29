import os
import httpx
from pynight.common_icecream import ic

# import logging
# logging.basicConfig(level=logging.DEBUG)


async def fetch_url(url: str) -> str:
    """
    Fetches the content of a URL. Uses HTTP_PROXY environment variable for proxy if set.

    Args:
    - url (str): The URL to fetch.

    Returns:
    - str: The content of the URL.

    Example usage:
        content = await fetch_url("https://www.example.com")
    """

    # Get proxy from environment variables
    http_proxy = os.environ.get("HTTP_PROXY")

    # If there's a proxy, use it. Otherwise, use default settings.
    # ic(http_proxy)
    if http_proxy:
        client_args = {
            "proxies": {"http://": http_proxy, "https://": http_proxy},
            # "verify": False,  # This disables SSL verification
        }
    else:
        client_args = {}

    async with httpx.AsyncClient(**client_args) as client:
        response = await client.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
