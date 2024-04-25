from openai import OpenAI

from brish import z, zp


deepseek_client = None
##
deepseek_key = None


def setup_deepseek_key():
    global deepseek_key
    global deepseek_client

    deepseek_key = z("var-get deepseek_api_key").outrs
    assert deepseek_key, "setup_deepseek_key: could not get Deepseek API key!"

    deepseek_client = OpenAI(
        # defaults to os.environ.get("DEEPSEEK_API_KEY")
        api_key=deepseek_key,
        base_url="https://api.deepseek.com/v1",
    )
    return deepseek_client


def deepseek_key_get():
    global deepseek_key

    if deepseek_key is None:
        setup_deepseek_key()

    return deepseek_key


##
