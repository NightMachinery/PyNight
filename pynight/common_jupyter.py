import json
import uuid
from IPython.display import display, display_javascript, HTML
import gc
from IPython import get_ipython
from .common_json import JSONEncoderWithFallback

##
def clipboard_copy_jupyter(obj, indent=4):
    json_encoder = JSONEncoderWithFallback(fallback_function=str, indent=indent)
    obj_json = json_encoder.encode(obj)
    obj_text = str(json.loads(obj_json))

    button_id = f"copy-btn-{uuid.uuid4()}"
    html_code = f"""
    <button id="{button_id}" style="padding: 10px; background-color: #4CAF50; border: none; color: white; cursor: pointer;">
        Copy to clipboard
    </button>
    """

    js_code = f"""
    function updateButton(button_id, message, color) {{
        const button = document.getElementById(button_id);
        button.style.backgroundColor = color || "#f44336";
        button.innerText = message;
    }}

    function copyToClipboard() {{
        updateButton("{button_id}", "Copying ...", "rebeccapurple");

        var textToCopy = {obj_json};
        if (typeof textToCopy !== 'string') {{
            try {{
                textToCopy = JSON.stringify(textToCopy, null, {indent});
            }} catch (err) {{
                console.error("Error stringifying some_var:", err);
                updateButton("{button_id}", "Error: " + err.message);
                return;
            }}
        }}

        navigator.clipboard.writeText(textToCopy)
            .then(() => {{
                updateButton("{button_id}", "Copied!", "#008CBA");
            }})
            .catch(err => {{
                console.error('Unable to copy text: ', err);
                updateButton("{button_id}", "Error: " + err.message);
            }});
    }}

    document.getElementById("{button_id}").addEventListener("click", copyToClipboard);
    """

    display(HTML(html_code))
    display_javascript(js_code, raw=True)

    return obj_text
##
def jupyter_gc():
    global_ns = get_ipython().user_ns

    del global_ns['Out']

    gc.collect()
##
