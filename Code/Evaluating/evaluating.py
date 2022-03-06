import re


def check_for_ngram(text1, text2, ngram=3):
    short_text = text1 if len(text1) <= len(text2) else text2
    long_text = text1 if len(text1) > len(text2) else text2

    words = clean_string(short_text).split()
    for i in range(len(words) - ngram + 1):
        ngram_words = ' '.join(words[i:i+ngram])
        if ngram_words in clean_string(long_text):
            return True
    return False


def clean_string(string):
    remove_regex = '[,.:\n\t\r]'
    clean =re.sub(remove_regex, ' ', string)
    return clean
