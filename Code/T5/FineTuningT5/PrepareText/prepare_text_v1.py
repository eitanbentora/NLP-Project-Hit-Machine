import re
import pandas as pd

version = 'v1'

def reformat_spaces(s, max_consecutive_spaces=100):
    s = re.sub('\r\n', '<n>', s)
    s = re.sub('\n', '<n>', s)
    s = re.sub('\t', '<t>', s)
    for sp in range(max_consecutive_spaces, 1, -1):
        s = re.sub(' '*sp, f'<s{sp}>', s)
    return s


df = pd.read_pickle('/home/student/Desktop/Project/Data/chords_en.pkl')

df['text'] =  df['song_name'] + '\n' + df['chords&lyrics']
df['text'] = df['text'].apply(lambda x: reformat_spaces(x))

artists_df = pd.read_pickle('/home/student/Desktop/Project/Data/artists.pkl')
df = df.merge(artists_df, left_on='artist_name', right_on='name', how='inner')
df = df[['song_name', 'text', 'genres', 'name']]

df.to_pickle(f'/home/student/Desktop/Project/Data/T5Data/PreparedText/prepared_data_{version}.pkl')

