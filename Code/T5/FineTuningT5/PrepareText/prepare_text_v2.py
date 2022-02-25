import re
import pandas as pd

version = 'v2'

def reformat_spaces(s, max_consecutive_spaces=100):
    s = re.sub('\r\n', ' @ ', s)
    s = re.sub('\n', ' @ ', s)
    return s


df = pd.read_pickle('/home/student/Desktop/Project/Data/chords_en.pkl')

df['lyrics_clean'] = df['lyrics'].apply(lambda x: '\n'.join([line for line in x.values()]))

df['text'] =  df['song_name'] + '\n' + df['lyrics_clean']
df['text'] = df['text'].apply(lambda x: reformat_spaces(x))


artists_df = pd.read_pickle('/home/student/Desktop/Project/Data/artists.pkl')
df = df.merge(artists_df, left_on='artist_name', right_on='name', how='inner')
df = df[['song_name', 'text', 'genres', 'name']]

df.to_pickle(f'/home/student/Desktop/Project/Data/T5Data/PreparedText/prepared_data_{version}.pkl')

