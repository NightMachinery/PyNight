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
            "\u0600" <= c <= "\u06FF"
            or "\u0750" <= c <= "\u077F"
            or "\u08A0" <= c <= "\u08FF"
        ):
            return True
    return False
