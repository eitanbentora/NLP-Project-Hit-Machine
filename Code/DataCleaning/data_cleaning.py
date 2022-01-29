from langdetect import detect


def predict_lang(text):
    try:
        lang = detect(text)
    except:
        print(text)
        lang = None
    return lang