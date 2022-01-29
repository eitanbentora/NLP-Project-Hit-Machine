import pandas as pd
import json
import warnings
import pickle
import glob
import web_scraping
from tqdm import tqdm

artists_df = web_scraping.prepare_artists_df('/home/student/Desktop/Project/Data/artists.csv')

# with open('/home/student/Desktop/Project/Data/ChordsData/missing.pkl', 'rb') as f:
#     missing_list = pickle.load(f)
    
save_dir = '/home/student/Desktop/Project/Data/ChordsData/MissingSongs/'

missing_artists_df = artists_df[artists_df['name_e_chords'].isin(['p!nk', 'ac/dc', "guns-n'-roses", "r.e.m.", "simon-&-garfunkel"])]
missing_artists_df['name_e_chords'] = pd.Series(['pink', 'acdc', 'guns-n-roses', 'rem', 'simon-garfunkel'], index = missing_artists_df.index)

web_scraping.create_chord_data(artists_df=missing_artists_df, save_dir=save_dir, save_every=1, artists_per_file=100)