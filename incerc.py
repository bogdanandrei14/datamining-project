import re

str = "marian george nu prea stie stie python. Www  weee w w w www  wd ew  w wwfdwdfww."

def tokenize(sentences):
    words = []
    wi = word_extraction(sentences)
    words.extend(wi)
    words = sorted(list(set(words)))
    return words


def word_extraction(sentence):
    words = re.sub("[^\w]", " ",  sentence).split()
    cleaned_text = []
    for w in words:
            if (w.isnumeric()) == False:
                if len(w) > 1:
                    if w.isalpha():
                        cleaned_text.append(w.lower())

    return cleaned_text

words = tokenize(str)
print(words)


