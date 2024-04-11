import base64
import tempfile
import os
from PIL import Image


##
def base64_encode_file(file_path):
    with open(file_path, "rb") as file_file:
        return base64.b64encode(file_file.read()).decode("utf-8")


def convert_to_jpeg_and_base64_encode(file_path, url_p=True):
    if os.path.exists(file_path):
        with Image.open(file_path) as img:
            if img.format.lower() not in ["jpg", "jpeg"]:
                #: I don't know if =format= is normalized first or not, so I took the safest approach above.

                rgb_im = img.convert(
                    "RGB"
                )  # Convert to RGB if necessary (e.g., RGBA, P mode)

                # Save the image to a temporary file
                with tempfile.NamedTemporaryFile(
                    suffix=".jpeg", delete=True
                ) as tmpfile:
                    temp_path = tmpfile.name
                    rgb_im.save(temp_path, format="JPEG")

                    # Use the temporary file for base64 encoding
                    file_base64 = base64_encode_file(temp_path)
            else:
                #: If the image is already a JPEG, encode it directly
                file_base64 = base64_encode_file(file_path)

            if url_p:
                return f"data:image/jpeg;base64,{file_base64}"
            else:
                return file_base64
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")


##
