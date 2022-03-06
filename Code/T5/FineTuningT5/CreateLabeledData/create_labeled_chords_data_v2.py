import pandas as pd
import ast
import sys
from chords_data_preparation import *

def create_labeled_df(chords_df, window=16):
    """expects a df with the columns: ['lyrics', 'train_chords', 'genres']"""
    input_ids = []
    labels = []
    for i, row in chords_df.iterrows():
        lyrics_dict = row['lyrics']
        chords_dict = row['train_chords']
        genres = sorted(ast.literal_eval(row['genres']))
        genres = '[' + ', '.join(genres) + ']'
        current_input = ['write chords ' + genres + ':']
        chords_lines = chords_dict.keys()
        current_chord_output = ''
        for line_idx in chords_lines:
            if line_idx + 1 in lyrics_dict.keys() and lyrics_dict[line_idx+1].replace(' ', '') != '':
                if current_chord_output:
                    current_input.insert(len(current_input)-1, current_chord_output)
                current_input.append(lyrics_dict[line_idx+1])
                current_chord_output = chords_dict[line_idx]
                input_ids.append(" @ ".join([current_input[0]]+current_input[-window:]))
                labels.append(current_chord_output)
    return pd.DataFrame({'input_ids': input_ids, 'labels':labels})

if __name__ ==  '__main__':
    version = 'v2'
    print('version: ', version)
    data = pd.read_pickle('/home/student/Desktop/Project/Data/chords_en.pkl')
    artists_df = pd.read_pickle('/home/student/Desktop/Project/Data/artists.pkl')
    chords_df = data.merge(artists_df, left_on='artist_name', right_on='name', how='left')
    chords_df = chords_df.drop_duplicates(subset=['chords&lyrics', 'artist_name', 'song_name'])
    print('finished merging dataframes')
    chords_df['train_chords'] = chords_df['chords'].apply(lambda x: prepare_chords_dict_for_train(x))
    print('finished parsing chords')
    labeled_chords_data = create_labeled_df(chords_df)
    print('finished creating labeled data')
    labeled_chords_data.to_pickle(f'/home/student/Desktop/Project/Data/T5Data/LabeledData/labeled_chords_data_{version}.pkl')
