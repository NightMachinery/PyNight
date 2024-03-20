import tiktoken
from types import SimpleNamespace
import openai
from openai import OpenAI
import os
import tempfile
from brish import z, zp
from pynight.common_base64 import (
    base64_encode_file,
    convert_to_jpeg_and_base64_encode,
)
from pynight.common_bells import bell_gpt
from pynight.common_dict import simple_obj
from pynight.common_str import (
    whitespace_shared_rm,
)
from pynight.common_clipboard import (
    clipboard_copy,
    clipboard_copy_multi,
)

##
openai_key = None
openai_client = None

def setup_openai_key():
    global openai_key
    global openai_client

    openai_key = z("var-get openai_api_key").outrs
    assert openai_key, "setup_openai_key: could not get OpenAI API key!"

    openai_client = OpenAI(api_key=openai_key)
    return openai_client


def openai_key_get():
    global openai_key

    if openai_key is None:
        setup_openai_key()

    return openai_key


###
import pyperclip
from icecream import ic
import subprocess
import time
import sys


def print_chat_streaming(
    output,
    *,
    debug_p=False,
    # debug_p=True,
    output_mode=None,
    copy_mode="chat2",
    # copy_mode="default",
    end="\n-------",
    backend="auto",
    # backend="OpenAI",
):
    """
    Process and print out chat completions from a model when the stream is set to True.

    Args:
        output (iterable): The output from the model with stream=True.
    """
    if backend == "auto":
        import anthropic

        if isinstance(output, anthropic.Stream):
            backend = "Anthropic"
        else:
            backend = "OpenAI"

    text = ""
    r = None
    for i, r in enumerate(output):
        text_current = None

        if backend == "OpenAI":
            if not isinstance(r, dict):
                #: OpenAI v1: Response objects are now pydantic models instead of dicts.
                ##
                r = dict(r)
                # ic(r)

            choice = r["choices"][0]
            choice = dict(choice)

            if "delta" in choice:
                delta = choice["delta"]
                delta = dict(delta)

                if i >= 1:
                    #: No need to start all responses with 'assistant:'.
                    ##
                    if "role" in delta and delta['role']:
                        if i >= 1:
                            print("\n", end="")

                        print(f"{delta['role']}: ", end="")

                if "content" in delta and delta['content']:
                    text_current = f"{delta['content']}"
            elif "text" in choice and choice['text']:
                text_current = f"{choice['text']}"

        elif backend == "Anthropic":
            if r.type == "content_block_start":
                text_current = r.content_block.text
            elif r.type == "content_block_delta":
                text_current = r.delta.text

        if text_current:
            text += text_current
            print(f"{text_current}", end="")

    print(end, end="")

    text = text.rstrip()

    if debug_p == True:
        ic(r)

    chat_result = None
    if copy_mode:
        chat_result = chatml_response_text_process(
            text,
            copy_mode=copy_mode,
        )

    if output_mode == "chat":
        return chat_result
    elif output_mode == "text":
        return text
    elif not output_mode:
        return None
    else:
        raise ValueError(f"Unsupported output_mode: {output_mode}")


def chatml_response_process(
    response,
    end="\n-------",
    **kwargs,
):
    for choice in response["choices"]:
        text = choice["message"]["content"]

        chatml_response_text_process(
            text,
            **kwargs,
        )
        print(text, end="")

        print(end, end="")


def chatml_response_text_process(
    text,
    copy_mode="chat2",
    # copy_mode="default",
):
    #: 'rawchat' is currently useless, just use 'text'.
    ##
    text_m = None
    if copy_mode in ("chat", "chat2"):
        text_m = f'''        {{"role": "assistant", "content": r"""{text}"""}},'''
    elif copy_mode in ("rawchat"):
        text_m = f"""{text}"""

    if copy_mode == "chat2":
        text_m += f'''
        {{"role": "user", "content": r"""\n        \n        """}},'''

    if copy_mode in (
        "default",
        "chat2",
    ):
        clipboard_copy_multi(text, text_m)

    elif copy_mode in (
        "chat",
        # "chat2",
        "rawchat",
    ):
        clipboard_copy(text_m)

    elif copy_mode == "text":
        clipboard_copy(text)

    return simple_obj(
        text=text,
        text_chat=text_m,
    )


def writegpt_process(messages_lst):
    out = ""
    seen = [
        "PLACEHOLDER",
    ]
    #: We can also just count the number of assistant outputs previously seen, and skip exactly that many. That way, we can edit the text more easily.

    for messages in messages_lst:
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role in ("assistant",) and content not in seen:
                seen.append(content)

                if out:
                    out += "\n\n"

                out += content

    out = subprocess.run(
        ["perl", "-CIOE", "-0777", "-pe", r"s/(*plb:\S)(\R)(*pla:\S)/\\\\$1/g"],
        text=True,
        input=out,
        errors="strict",
        encoding="utf-8",
        capture_output=True,
    ).stdout

    clipboard_copy(out)
    return out


def openai_chat_complete(
    *args,
    model="gpt-3.5-turbo",
    messages=None,
    stream=True,
    interactive=False,
    copy_last_message=None,
    trim_p=True,
    # backend="OpenAI",
    backend="auto",
    **kwargs,
):
    print("/❂\\") #: to detect dead kernels

    model_orig = model

    if backend == "auto":
        if "claude" in model_orig.lower():
            backend = "Anthropic"
        else:
            backend = "OpenAI"

    if model_orig in (
        "gpt-4-turbo-auto-vision",
        "gpt-4-turbo",
        "4t",
    ):
        # model = "gpt-4-1106-preview"
        model = "gpt-4-0125-preview"

    def clean_message(msg):
        msg = whitespace_shared_rm(msg)
        msg = msg.strip()

        return msg

    if interactive:
        if copy_last_message is None:
            copy_last_message = True

    system_message = None
    if messages is not None:
        messages_processed = []

        for message in messages:
            if isinstance(message["content"], str):
                if trim_p:
                    message["content"] = clean_message(message["content"])
            elif isinstance(message["content"], list):
                for i, msg in enumerate(message["content"]):
                    if isinstance(msg, dict) and "type" in msg:
                        if msg["type"] == "text":
                            if trim_p:
                                msg["text"] = clean_message(msg["text"])
                        elif msg["type"] == "image_url":
                            if model_orig in ["gpt-4-turbo-auto-vision"]:
                                model = "gpt-4-vision-preview"

            if backend == "Anthropic":
                if "role" in message and message["role"] == "system":
                    assert system_message is None, "Only one system message is allowed for Anthropic."

                    system_message = message["content"]
                    message = None

            if message is not None:
                messages_processed.append(message)

        messages = messages_processed

    try:
        while True:
            if copy_last_message:
                last_message = messages[-1]["content"]
                if isinstance(last_message, str):
                    clipboard_copy(last_message)

            if backend == "OpenAI":
                try:
                    return openai_client.chat.completions.create(*args,
                    model=model,
                    messages=messages,
                    stream=stream,
                    **kwargs)

                except openai.RateLimitError:
                    print(
                        "OpenAI ratelimit encountered, sleeping ...",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(10)  #: in seconds

            elif backend == "Anthropic":
                import anthropic
                from pynight.common_anthropic import (
                    anthropic_client,
                )

                if system_message:
                    assert "system" not in kwargs, "Only one system message is allowed for Anthropic."
                    kwargs["system"] = system_message

                return anthropic_client.messages.create(
                    *args,
                    model=model,
                    messages=messages,
                    stream=stream,
                    **kwargs,
                )
    finally:
        pass


###
def truncate_by_tokens(text, length=3500, model="gpt-3.5-turbo"):
    #: @deprecated?
    #: @alt =ttok=
    ##
    encoder = tiktoken.encoding_for_model(model)

    encoded = encoder.encode(text)

    truncate_p = len(encoded) > length
    encoded_rest = None
    if truncate_p:
        encoded_truncated = encoded[:length]
        encoded_rest = encoded[length:]
    else:
        encoded_truncated = encoded

    text_truncated = encoder.decode(encoded_truncated)
    text_rest = None
    if encoded_rest:
        text_rest = encoder.decode(encoded_rest)

    return SimpleNamespace(
        text=text_truncated,
        text_rest=text_rest,
        truncated_p=truncate_p,
    )


###
def openai_image_url_auto(
    url,
    strip_p=True,
    magic_p=True,
):
    if strip_p:
        url = url.strip()

    if magic_p:
        if url in ("clip", "MAGIC_CLIPBOARD"):
            format = "jpg"
            #: We need to convert to JPEG later anyway.

            with tempfile.NamedTemporaryFile(
                suffix=f".{format}", delete=False
            ) as tmpfile:

                if format == "jpg":
                    z("jpgpaste {tmpfile.name}").assert_zero
                elif format == "png":
                    z("pngpaste {tmpfile.name}").assert_zero
                else:
                    raise ValueError(f"Unsupported format: {format}")

                url = tmpfile.name

    if os.path.exists(url):
        file_base64 = convert_to_jpeg_and_base64_encode(
            url,
            url_p=True,
        )
        return file_base64
    else:
        return url


##
