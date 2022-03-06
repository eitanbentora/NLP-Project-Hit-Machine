import pandas as pd
import ast
from chords_data_preparation import *  

def create_labeled_df(chords_df):
    """expects a df with the columns: ['lyrics', 'train_chords', 'genres']"""
    input_ids = []
    labels = []
    for i, row in chords_df.iterrows():
        lyrics_dict = row['lyrics']
        chords_dict = row['train_chords']
        genres = sorted(ast.literal_eval(row['genres']))
        genres = '[' + ', '.join(genres) + ']'
        song = 'write chords ' + genres + ': '
        chords = ''
        chords_lines = chords_dict.keys()
        for line_idx in chords_lines:
            if line_idx + 1 in lyrics_dict.keys() and lyrics_dict[line_idx+1].replace(' ', '') != '':
                chords += chords_dict[line_idx] + ' @ '
                song += lyrics_dict[line_idx+1] + ' @ '
        input_ids.append(song)
        labels.append(chords)

    return pd.DataFrame({'input_ids': input_ids, 'labels':labels})

if __name__ ==  '__main__':
    version = 'v1'
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
