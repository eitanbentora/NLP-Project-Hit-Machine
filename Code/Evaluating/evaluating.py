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
    clean = ' '.join(clean.split())
    return clean

def make_n_gram_dict(df, lyrics, min_ngram=4):
    n_grams_match = {}
    ngram = min_ngram
    continue_run = True
    while continue_run:
        print(ngram)
        n_grams_match[ngram] = {}
        continue_run = False
        for i, lyric in enumerate(lyrics):
            n_grams_match[ngram][f'song_{i}'] = df['lyrics2'].apply(lambda x: check_for_ngram(lyric, x, ngram=ngram)).sum()
            if n_grams_match[ngram][f'song_{i}'] > 0:
                continue_run = True
        ngram += 1
        if ngram == 12:
            break
    return n_grams_match