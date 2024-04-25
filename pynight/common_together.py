from openai import OpenAI

from brish import z, zp


together_client = None
##
together_key = None


def setup_together_key(
    backend="openai", #: less buggy, no need for another package
    # backend="native",
):
    global together_key
    global together_client

    together_key = z("var-get together_api_key").outrs
    assert together_key, "setup_together_key: could not get Together API key!"

    if backend == "native":
        from together import Together

        together_client = Together(
            api_key=together_key,
        )

    elif backend == "openai":
        together_client = OpenAI(
            api_key=together_key,
            base_url="https://api.together.xyz/v1",
        )

    return together_client


def together_key_get():
    global together_key

    if together_key is None:
        setup_together_key()

    return together_key


##
