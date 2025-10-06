##
#: @duplicateCode/5a0df91917ef00a5024c8918aa56b273

persian_chars = (
    "ضصثقفغعهخحجچشسیبلاتنمکگظطزرذدپو.ًٌٍَُِّْ][}{|ؤئيإأآة»«:؛كٓژٰ‌ٔء<>؟٬٫﷼٪×،)(ـ۱۲۳۴۵۶۷۸۹۰"
)
en_chars = "qwertyuiop[]asdfghjkl;'zxcvbnm,.QWERTYUIOP{}|ASDFGHJKL:\"ZXCVBNM<>?@#$%^&()_1234567890"
#: Note that Zsh escaping is different from Python; in particular, `\$` should become `$`.

# Create a translation table
# str.maketrans maps each character in the first string to the character
# at the corresponding position in the second string.
#: [[https://docs.python.org/3/library/stdtypes.html#:~:text=static-,str.maketrans][Built-in Types — Python 3.13.7 documentation]]
per2en_translation_table = str.maketrans(persian_chars, en_chars)
en2per_translation_table = str.maketrans(en_chars, persian_chars)


def per2en(input_str: str) -> str:
    """
    Translates Persian characters and numerals in the input string to their
    corresponding English/Latin counterparts based on a predefined mapping.

    Args:
        input_str: The string containing Persian characters to be translated.

    Returns:
        A new string with Persian characters replaced by English/Latin characters.
    """
    # Apply the translation to the input string
    return input_str.translate(per2en_translation_table)


def en2per(input_str: str) -> str:
    """
    Translates English/Latin characters and numerals in the input string to their
    corresponding Persian counterparts based on a predefined mapping.

    Args:
        input_str: The string containing English/Latin characters to be translated.

    Returns:
        A new string with English/Latin characters replaced by Persian characters.
    """
    # Apply the translation to the input string
    return input_str.translate(en2per_translation_table)


##
def rtl_reshaper_v1(text):
    import arabic_reshaper
    from bidi.algorithm import get_display

    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text


def rtl_reshaper_fribidi(text):
    raise NotImlpementedError()


def contains_persian(text):
    #: @o1
    ##
    for c in text:
        if (
            "\u0600" <= c <= "\u06ff"
            or "\u0750" <= c <= "\u077f"
            or "\u08a0" <= c <= "\u08ff"
        ):
            return True
    return False
