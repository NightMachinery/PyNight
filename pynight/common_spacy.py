import spacy

##
nlp_sentencizer = spacy.blank("en")
nlp_sentencizer.add_pipe("sentencizer")


def spacy_sentencizer(text):
    tokens = nlp_sentencizer(text)
    text_s = [str(sent) for sent in tokens.sents]
    return text_s


##
nlp_sentencizer_fa = spacy.blank("fa")
nlp_sentencizer_fa.add_pipe("sentencizer")


def spacy_sentencizer_fa(text):
    tokens = nlp_sentencizer(text)
    text_s = [str(sent) for sent in tokens.sents]
    return text_s


##
