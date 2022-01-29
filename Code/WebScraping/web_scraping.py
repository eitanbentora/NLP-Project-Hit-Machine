from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
from tqdm import tqdm
import warnings
import pickle
import glob
import unidecode
import os.path

    
def get_song_links_from_artist_url(artist_url):
    artist_result = requests.get(artist_url)
    artist_doc = BeautifulSoup(artist_result.text, 'html.parser')
    artist_song_urls_dict = {}
    for h2_tag in artist_doc.find('ul', {'id': 'results'}).find_all('h2'):
        song_url = h2_tag.find('a', href=True)['href']
        song_name = h2_tag.find('a', href=True).text
        artist_song_urls_dict[song_name] = song_url
    return artist_song_urls_dict


def get_chords_from_song_url(song_url):
#     song_url = unidecode.unidecode(song_url, errors='replace')
    try:
        song_result = requests.get(song_url)
    except UnicodeDecodeError as e: 
        print(e)
        return None
        
    song_doc = BeautifulSoup(song_result.text, 'html.parser')
    song_chords = song_doc.find("div",class_= "coremain")
    if song_chords is not None:
        return song_chords.text
    return None


def get_song_chords_of_an_artist(artist_url, start_from=0, end_at=float('inf')):
    assert start_from < end_at
    artist_song_urls_dict = get_song_links_from_artist_url(artist_url)
    artist_song_chords_list = []
    for i, (song_name, song_url) in enumerate(artist_song_urls_dict.items()):
        if i < start_from:
            continue
        if i > end_at:
            break
        chords = get_chords_from_song_url(song_url)
        if chords is not None:
            artist_song_chords_list.append({'name': song_name, 'chords': chords})
        else:
            print(f"song {song_url} is empty")
    return artist_song_chords_list

        

def save_files(save_dir, missing_list, json_data, file_ending):
    with open(save_dir + 'missing.pkl', 'wb') as f:
        pickle.dump(missing_list, f)

    with open(save_dir + f'chords_{file_ending}.json', 'w') as f:
        json.dump(json_data, f)

        
def prepare_artists_df(path):
    artist_df = pd.read_csv(path)
    artist_df = artist_df.sort_values('popularity', ascending=False)
    artist_df['name_e_chords'] = artist_df['name'].apply(lambda x: x.lower().replace(' ', '-'))
    artist_df = artist_df.reset_index(drop=True)
    return artist_df


def get_last_file_ending_num(file_path):
    file_ending_num = int(file_path.split('_')[1].replace('.json', ''))
    return file_ending_num


def get_last_file(save_dir):
    json_files = glob.glob(save_dir + '*.json')
    json_files = sorted(json_files, key=lambda x: int(x.split('_')[1].replace('.json', '')))
    return json_files[-1]
    

def get_last_artist_index(last_json_file, artists_df):
    with open(last_json_file) as f:
        chords_data = json.load(f)
    last_artist_name  = chords_data['artists'][-1]['name_spotify']
    last_artist_index = artists_df[artists_df['name'] == last_artist_name].index[0]
    return last_artist_index


def create_chord_data(artists_df, save_dir, save_every, artists_per_file, songs_per_artist=float('inf')):
    if isinstance(artists_df, str):
        artists_df = prepare_artists_df(artists_df_path)
    base_url = "https://www.e-chords.com/chords/"
    json_data = {"artists": []}
    
    missing_file = save_dir + 'missing.pkl'
    
    if os.path.isfile(missing_file):
        with open(missing_file, 'rb') as f:
            missing_list = pickle.load(f)
        print(f'{len(missing_list)=}')
    else:
        missing_list = []
    
    if os.path.isfile(save_dir + 'chords_0.json'):
        last_file = get_last_file(save_dir)
        file_ending = get_last_file_ending_num(last_file) + 1
        artist_index = get_last_artist_index(last_file, artists_df) + 1

        print(f'{file_ending=}')
        print(f'{artist_index=}')
    else:
        file_ending = 0
        artist_index = 0
        
    artist_count = 0
    for i, row in tqdm(artists_df.iterrows(), total=len(artists_df)):
        if i <= artist_index:
            continue
        
        artist_url = base_url + row['name_e_chords']
        try:
            songs = get_song_chords_of_an_artist(artist_url, end_at=songs_per_artist)

        except AttributeError:
            print(f'did not find artist {artist_url}')
            missing_list.append(row['name_e_chords'])
            continue
        json_data['artists'].append(
            {
                'name_e_chords': row['name_e_chords'],
                'genres': row['genres'],
                'name_spotify': row['name'],
                'popularity': row['popularity'],
                'followers': row['followers'],
                'index': i,
                'songs': songs
            })

        if artist_count % save_every == 0:
            save_files(save_dir, missing_list, json_data, file_ending)

        if artist_count % artists_per_file == 0 and artist_count != 0:
            save_files(save_dir, missing_list, json_data, file_ending)
            json_data = {"artists": []}
            file_ending += 1
            
        artist_count += 1

        
        
def load_json_files(save_dir, load_last_file=True):
    json_files = glob.glob(save_dir + '*.json')
    json_files = sorted(json_files, key=lambda x: int(x.split('_')[1].replace('.json', '')))
    if not load_last_file:
        json_files = json_files[:-1]
    artists = []
    for json_file in json_files:
        with open(json_file) as f:
            chords_data = json.load(f)
            artists += chords_data['artists']
    return {'artists': artists}

def json_to_df(chords_json):
    df_dict = {'artist_name': [], 'song_name': [], 'chords': []}
    for artist in chords_json['artists']:
        df_dict['artist_name'] += [artist['name_spotify']]*len(artist['songs'])
        for song in artist['songs']:
            df_dict['song_name'].append(song['name'])
            df_dict['chords'].append(song['chords'])

    return pd.DataFrame(df_dict)

        

if __name__ == '__main__':
    save_dir = '/home/student/Desktop/Project/Data/ChordsData/'

    artists_df_path = '/home/student/Desktop/Project/Data/artists.csv'

    create_chord_data(artists_df=artists_df_path, save_dir=save_dir, save_every=5, artists_per_file=100)


    # with open(save_dir + 'chords_1.json') as json_file:
    #     chords_data = json.load(json_file)
    #
    #
    # with open(save_dir + 'missing.pkl', 'rb') as f:
    #     missing_list = pickle.load(f)


