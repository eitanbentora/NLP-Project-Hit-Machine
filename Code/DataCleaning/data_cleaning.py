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
    chord_regex = r"^\(?([ABCDEFG])([#b]?)(m?)-?(\(?[245679]?\)?)(\-?)(/?)((dim)|(sus)|(maj)|(aug)|)(\+?)(add)?(\(?([245679]|11|13)?\)?)M?(\*?)((/[ABCDEFG][#b]?)?)(\(hold\))?\)?$"
    return bool(re.match(chord_regex, word)) and word != ''


def is_chord_line(line):
    special_words= ['intro', 'x2', 'solo', 'interlude', 'break', '%', 'bridge', 'introdução']
    
    if line.lower().startswith('bridge:'):
        return True
    
    if len(line) == 0:
        return False
    
    chord_count, word_count = 0, 0
    
    for word in re.split(r' |,|\||:|\(|\)| -| -|\t|\n',line):
        if word != '':
            word_count += 1
        if is_chord(word) or word.lower() in special_words:
            chord_count += 1
    if word_count == 0:
        return False
    if chord_count/word_count >= 1/2:
        return True
    return False

def is_tab_line(line):
    if 'hide this tab' in line.lower():
        return True
    if line.count('-') > 4:
        return True
    tab_regex = r"^ *([ABCDEFGabcde]?)([#b]?)(\|*)([\d\-\^><\|]+)( *)(\|*) *$"
    return bool(re.match(tab_regex, line)) and len(line) > 0
    

def remove_line(line):
    if re.match(' *^_+ *$', line):
        return True
    remove = ['capo on']
    for word in re.split(r'[ +]',line):
        if word.lower() in remove:
            return True
    return False
    

def split_chords_lyrics_and_tabs(song):
    lines = [x for x in song.split('\r\n')]
    chords, lyrics, tabs = {}, {}, {}
    for i, line in enumerate(lines):  
        if remove_line(line):
            continue
        elif is_chord_line(line):
            chords[i] = line
        elif is_tab_line(line):
            tabs[i] = line
        else:
            lyrics[i] = line
    return chords, lyrics, tabs
    