import io

def cat(file_path):
    with io.open(file_path, 'r', encoding='utf8') as file_:
        text = file_.read()
        return text
