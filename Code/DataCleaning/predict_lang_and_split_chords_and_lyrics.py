import pandas as pd
import data_cleaning as data_cleaning
from tqdm import tqdm
tqdm.pandas()

chords_df = pd.read_pickle('/home/student/Desktop/Project/Data/chords.pkl')
print("spliting")
chords_df[['chords', 'lyrics', 'tabs']] = chords_df[['chords&lyrics']].progress_apply(
    lambda x: data_cleaning.split_chords_lyrics_and_tabs(x[0]), axis=1, result_type ='expand')
print("pedicting lang")
chords_df['lang'] = chords_df['lyrics'].progress_apply(lambda x: '\n'.join(x.values()))\
                                            .progress_apply(lambda x: data_cleaning.predict_lang(x))

chords_df.to_pickle('/home/student/Desktop/Project/Data/chords_split_lang.pkl')