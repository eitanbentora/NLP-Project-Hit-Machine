import pandas as pd
import ast
import sys
sys.path.insert(1, '/home/student/Desktop/Project/Code/DataCleaning/')

from data_cleaning import is_chord 

    
def prepare_chord_line_for_train(chord_line, chord_separator='$'):
    chord_line = chord_line.replace('\t', ' '*4) + ' '
    train_chord_line = ''
    chord = ''
    num_spaces = 0
    for i, char in enumerate(chord_line):
        if char not in [' ', '\n', '\r\n']:
            chord += char
        else:
            if chord and is_chord(chord):
                train_chord_line += str(num_spaces-len(chord)) + chord + chord_separator
                num_spaces = 0
                chord = ''
            else:
                chord = ''
        num_spaces += 1
    return train_chord_line.strip(chord_separator).strip('0')


def prepare_chords_dict_for_train(chords_dict):
    return {key: prepare_chord_line_for_train(line) for key, line in chords_dict.items()}
