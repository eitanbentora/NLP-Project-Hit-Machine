from langdetect import detect
import pandas as pd
import itertools
import re

def predict_lang(text):
    try:
        lang = detect(text)
    except:
        print(text)
        lang = None
    return lang

def is_chord(word):
    chord_regex = r"^\(?([ABCDEFG])([#b]?)(m?)(\(?[245679]?\)?)(\-?)(/?)((dim)|(sus)|(maj)|(aug)|)(\+?)(add)?(\(?[245679]?\)?)M?(\*?)((/[ABCDEFG][#b]?)?)(\(hold\))?\)?$"
    return bool(re.match(chord_regex, word)) and word != ''

def is_chord_line(line):
    special_words= ['Intro:', 'X2' , 'x2', 'Intro', 'Solo', '2X)(', '(2X)', '(2x)', 'Interlude:', '(', ')', 'break', '(BREAK)']
    if len(line) == 0:
        return False
    is_chord_line=True
    for word in re.split(r'[ +]',line):
        if not is_chord(word) and word not in special_words and word != '':
            return False
    return True

def is_tab_line(line):
    tab_regex = r"^([ABCDEFGabcde]?)([#b]?)(\|*)([\d\-/\^><\|]+)( *)(\|*) *$"
    return bool(re.match(tab_regex, line)) and len(line) > 0


def split_chords_lyrics_and_tabs(song):
    lines = [x for x in song.split('\r\n')]
    chords, lyrics, tabs = {}, {}, {}
    for i, line in enumerate(lines):      
        if is_tab_line(line):
            tabs[i] = line
        elif is_chord_line(line):
            chords[i] = line
        else:
            lyrics[i] = line
    return chords, lyrics, tabs
    