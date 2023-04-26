import json
import uuid
from IPython.display import display, display_javascript, HTML


def clipboard_copy_jupyter(text: str):
    escaped_text = json.dumps(text)
    button_id = f"copy-btn-{uuid.uuid4()}"
    html_code = f"""
    <button id="{button_id}" style="padding: 10px; background-color: #4CAF50; border: none; color: white; cursor: pointer;">
        Copy to clipboard
    </button>
    """

    js_code = f"""
    function copyToClipboard() {{
        const textToCopy = {escaped_text};

        navigator.clipboard.writeText(textToCopy)
            .then(() => {{
                const button = document.getElementById("{button_id}");
                button.style.backgroundColor = "#008CBA";
                button.innerText = "Copied!";
            }})
            .catch(err => {{
                console.error('Unable to copy text: ', err);

                const button = document.getElementById("{button_id}");
                button.style.backgroundColor = "#f44336";
                button.innerText = "Error: " + err.message;
            }});
    }}

    document.getElementById("{button_id}").addEventListener("click", copyToClipboard);
    """

    display(HTML(html_code))
    display_javascript(js_code, raw=True)
