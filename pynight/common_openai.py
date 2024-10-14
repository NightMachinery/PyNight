import tiktoken
from types import SimpleNamespace
import openai
from openai import OpenAI
import os
import tempfile
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
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
from pynight.common_debugging import fn_name_current


##
#: @duplicateCode/1eda4a3f17ea02a0fca9b6d7a6b16663
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


##
#: @duplicateCode/1eda4a3f17ea02a0fca9b6d7a6b16663
openrouter_key = None
openrouter_client = None


def setup_openrouter_key():
    global openrouter_key
    global openrouter_client

    openrouter_key = z("var-get openrouter_api_key").outrs
    assert openrouter_key, "setup_openrouter_key: could not get OpenRouter API key!"

    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )
    return openrouter_client


def openrouter_key_get():
    global openrouter_key

    if openrouter_key is None:
        setup_openrouter_key()

    return openrouter_key


gemini_key = None
gemini_client = None


def setup_gemini_key():
    global gemini_key
    global gemini_client

    gemini_key = z("var-get gemini_api_key").outrs
    assert gemini_key, "setup_gemini_key: could not get Gemini API key!"

    genai.configure(api_key=gemini_key)
    gemini_client = genai
    return gemini_client


def gemini_key_get():
    global gemini_key

    if gemini_key is None:
        setup_gemini_key()

    return gemini_key


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
    stream_p=True,
    #: Detecting stream_p automatically might be challenging.
):
    """
    Process and print out chat completions from a model when the stream is set to True.

    Args:
        output (iterable): The output from the model with stream=True.
    """
    try:
        if backend == "auto":
            import anthropic

            if isinstance(output, anthropic.Stream):
                backend = "Anthropic"
            elif isinstance(output, genai.types.GenerateContentResponse):
                backend = "Gemini"
            else:
                backend = "OpenAI"

        if backend == "OpenRouter":
            backend = "OpenAI"
            #: should be the same for our purposes here

        text = ""

        #: Handle non-streaming responses
        if not stream_p:
            if backend == "OpenAI":
                text = output.choices[0].message.content

            ##
            #: Other backends not yet tested, LLM generated:
            elif backend == "Anthropic":
                text = output.content
            elif backend == "Gemini":
                text = output.text
            ##

            print(text, end=end)

        else:
            r = None
            last_role = None
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

                        if "role" in delta and delta["role"]:
                            if last_role is None:
                                last_role = delta["role"]
                            elif last_role != delta["role"]:
                                last_role = delta["role"]

                                print("\n", end="")
                                print(f"{delta['role']}: ", end="")

                        if "content" in delta and delta["content"]:
                            text_current = f"{delta['content']}"
                    elif "text" in choice and choice["text"]:
                        text_current = f"{choice['text']}"

                elif backend == "Anthropic":
                    if r.type == "content_block_start":
                        text_current = r.content_block.text
                    elif r.type == "content_block_delta":
                        text_current = r.delta.text

                elif backend == "Gemini":
                    text_current = r.text

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
    finally:
        if hasattr(output, "close"):
            output.close()
        #: The hope is to stop the upstream credit charges.
        #: [[id:fba91d52-7694-4894-8ab0-44d16aa96a90][Stream Cancellation]]

        # print(f"\n{fn_name_current()}: closed the connection")
        # ic(type(output))
        #: ic| type(output): <class 'openai.Stream'>


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


START_SYMBOL = "/❂\\"


openai_compatible_backends = [
    "OpenAI",
    "OpenRouter",
    "Groq",
    "DeepSeek",
    "Together",
]


def select_backend(model_orig):
    model_orig_normalized = model_orig.lower()

    if model_orig_normalized.startswith("or:"):
        backend = "OpenRouter"
        model = model_orig_normalized[3:]
    elif model_orig_normalized.startswith("gq:"):
        backend = "Groq"
        model = model_orig_normalized[3:]
    elif model_orig_normalized.startswith("tg:"):
        backend = "Together"
        model = model_orig_normalized[3:]
    elif model_orig_normalized.startswith("gai:"):
        backend = "Gemini"
        model = model_orig_normalized[4:]
    elif "deepseek" in model_orig_normalized:
        backend = "DeepSeek"
        model = model_orig_normalized
    elif "claude" in model_orig_normalized:
        backend = "Anthropic"
        model = model_orig_normalized
    else:
        backend = "OpenAI"
        if model_orig_normalized in (
            "gpt-4-turbo-auto-vision",
            "gpt-4-turbo",
            "4t",
        ):
            model = "gpt-4-turbo"
        else:
            model = model_orig_normalized

    return backend, model


def get_client(backend):
    if backend == "OpenAI":
        return openai_client
    elif backend == "OpenRouter":
        return openrouter_client
    elif backend == "Groq":
        from pynight.common_groq import groq_client

        return groq_client
    elif backend == "DeepSeek":
        from pynight.common_deepseek import deepseek_client

        return deepseek_client
    elif backend == "Together":
        from pynight.common_together import together_client

        return together_client
    elif backend == "Anthropic":
        from pynight.common_anthropic import anthropic_client

        return anthropic_client
    elif backend == "Gemini":
        return gemini_client
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def handle_ratelimit():
    print(
        "Ratelimit encountered, sleeping ...",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(10)  #: in seconds


def clean_message(msg):
    if msg:
        msg = whitespace_shared_rm(msg)
        msg = msg.strip()
    return msg


def openai_chat_complete(
    *args,
    model="gpt-3.5-turbo",
    messages=None,
    stream=True,
    interactive=False,
    copy_last_message=None,
    trim_p=True,
    backend="auto",
    system_repeat_mode="concat",
    **kwargs,
):
    print(START_SYMBOL)  #: to detect dead kernels
    #: The above marker needs to be excluded in [help:night/org-babel-result-get].

    model_orig = model
    backend, model = select_backend(model_orig)

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
                                # model = "gpt-4-vision-preview"
                                model = "gpt-4-turbo"

            if backend in ["Anthropic", "Gemini"]:
                if "role" in message and message["role"] == "system":
                    if system_repeat_mode == "error":
                        assert (
                            system_message is None
                        ), f"Only one system message is allowed for {backend}."

                    system_message = (system_message or "") + "\n" + message["content"]
                    message = None

            if message is not None:
                messages_processed.append(message)

        messages = messages_processed

    if trim_p:
        system_message = clean_message(system_message)

    try:
        while True:
            if copy_last_message:
                last_message = messages[-1]["content"]
                if isinstance(last_message, str):
                    clipboard_copy(last_message)

            if backend in openai_compatible_backends:
                client = get_client(backend)

                try:
                    response = client.chat.completions.create(
                        *args,
                        model=model,
                        messages=messages,
                        stream=stream,
                        **kwargs,
                    )
                    return response

                except openai.RateLimitError:
                    handle_ratelimit()

            elif backend == "Anthropic":
                client = get_client(backend)

                if system_message:
                    if system_repeat_mode == "error":
                        assert (
                            "system" not in kwargs
                        ), "Only one system message is allowed for Anthropic."
                    else:
                        if "system" in kwargs:
                            system_message = kwargs.system + "\n" + system_message

                    if trim_p:
                        system_message = clean_message(system_message)

                    kwargs["system"] = system_message

                return client.messages.create(
                    *args,
                    model=model,
                    messages=messages,
                    stream=stream,
                    **kwargs,
                )
            elif backend == "Gemini":
                client = get_client(backend)

                generation_config = {
                    "temperature": kwargs.get("temperature", 0),
                    "top_p": kwargs.get("top_p", 0.95),
                    "top_k": kwargs.get("top_k", 40),  #: topk
                    "max_output_tokens": kwargs.get("max_tokens", 8192),
                    "response_mime_type": "text/plain",
                }

                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }

                history = []
                for message in messages:
                    if message["role"] != "system":
                        role = (
                            "model"
                            if message["role"] == "assistant"
                            else message["role"]
                        )
                        history.append({"role": role, "parts": message["content"]})

                gemini_model_args = {
                    "model_name": model,
                    "generation_config": generation_config,
                    "safety_settings": safety_settings,
                }

                if system_message:
                    gemini_model_args["system_instruction"] = system_message

                gemini_model = client.GenerativeModel(**gemini_model_args)

                chat_session = gemini_model.start_chat(history=history)

                last_message = messages[-1]
                response = chat_session.send_message(last_message["content"])

                # ic(type(response))
                #: ic| type(response): <class 'google.generativeai.types.generation_types.GenerateContentResponse'>

                if stream:
                    #: Does this work for streaming?
                    return response

                else:
                    return SimpleNamespace(text=response.text)
            else:
                raise ValueError(f"Unsupported backend: {backend}")
    finally:
        pass


def openai_text_complete(
    *args,
    model="text-davinci-003",
    prompt=None,
    stream=True,
    echo=False,  #: Echo back the prompt in addition to the completion
    max_tokens=100,
    backend="auto",
    trim_p=True,
    **kwargs,
):
    print(START_SYMBOL)  #: to detect dead kernels
    #: The above marker needs to be excluded in [help:night/org-babel-result-get].

    model_orig = model
    backend, model = select_backend(model_orig)

    if trim_p:
        prompt = clean_message(prompt)

    try:
        while True:
            if backend in openai_compatible_backends:
                client = get_client(backend)

                try:
                    response = client.completions.create(
                        *args,
                        model=model,
                        prompt=prompt,
                        stream=stream,
                        echo=echo,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    return response

                except openai.RateLimitError:
                    handle_ratelimit()
            else:
                raise ValueError(f"Unsupported backend: {backend}")
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
