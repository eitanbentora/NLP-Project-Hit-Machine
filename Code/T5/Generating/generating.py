from tqdm import tqdm
import pickle
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import pandas as pd
from typing import List
import re
from generating_consts import LYRICS_GENERATING_PARAMS, CHORDS_GENERATING_PARAMS, CHORDS_WINDOW, LYRICS_PREFIX, \
    CHORDS_PREFIX, LYRICS_BEST_VERSION, CHORDS_BEST_VERSION


def decode_chord_line(encoded_chord_line):
    decoded_chord_line = ''
    for chord in encoded_chord_line.split('$'):
        chord = re.sub('[\(\)]', '', chord)
        if len(chord) > 0 and chord[0].isdigit():
            n_spaces = re.search(r'\d+', chord).group()
            l_spaces, n_spaces = len(n_spaces), int(n_spaces)
        else:
            l_spaces, n_spaces = 0, 0
        decoded_chord_line += n_spaces*' '
        decoded_chord_line += chord[l_spaces:]
    return decoded_chord_line


class SongWriter:
    def __init__(self, lyrics_model, chords_model, tokenizer, lyrics_prefix=LYRICS_PREFIX,
                 chords_prefix=CHORDS_PREFIX, verbose=True):
        self.lyrics_prefix = lyrics_prefix
        self.chords_prefix = chords_prefix
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lyrics_model = lyrics_model.to(device=self.device)
        self.chords_model = chords_model.to(device=self.device)
        self.tokenizer = tokenizer
        self.verbose = verbose

    def print(self, msg):
        if self.verbose:
            print(msg)

    def prepare_lyrics_inputs(self, genres_list, song_inputs):
        if isinstance(song_inputs, str):
            song_inputs = [song_inputs]

        if isinstance(genres_list, str):
            genres_list = [[genres_list]]

        if isinstance(genres_list[0], str) and len(song_inputs) == 1:
            genres_list = [genres_list]

        if isinstance(genres_list[0], str) and 1 < len(song_inputs) == len(genres_list):
            genres_list = [[g] for g in genres_list]

        assert len(song_inputs) == len(genres_list)
        assert isinstance(genres_list[0][0], str) and isinstance(song_inputs[0], str)

        song_inputs_list = []
        for genres, song_input in zip(genres_list, song_inputs):
            song_inputs_list.append(f"{self.lyrics_prefix} [{', '.join(sorted(genres))}]: {song_input}")
        return song_inputs_list

    def write_lyrics(self, genres_list: List[List[str]], song_inputs: List[str], lyrics_generating_params=None, batch_size=None):
        if not lyrics_generating_params:
            lyrics_generating_params = LYRICS_GENERATING_PARAMS
        print(LYRICS_GENERATING_PARAMS)
        song_inputs_list = self.prepare_lyrics_inputs(genres_list, song_inputs)
        if not batch_size:
            batch_size = len(song_inputs_list)
        self.print(song_inputs_list)
        lyrics = []
        for i in range(0, len(song_inputs), batch_size):
            inputs = self.tokenizer(song_inputs_list[i:i+batch_size], return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.lyrics_model.generate(
                    input_ids=inputs['input_ids'].to(device='cuda'),
                    attention_mask=inputs['attention_mask'].to(device='cuda'),
                    **lyrics_generating_params
                )
            lyrics += self.tokenizer.batch_decode(outputs.to(device='cpu'), skip_special_tokens=True)
        return [x.replace('@', '\n').replace("'?", "'") for x in lyrics]

    @staticmethod
    def fix_write_chords_params(genres_list, lyrics_list):
        if isinstance(genres_list, str):
            genres_list = [[genres_list]]
        if isinstance(lyrics_list, str):
            lyrics_list = [lyrics_list]
        if isinstance(genres_list[0], str) and len(lyrics_list) == 1:
            genres_list = [genres_list]
        if isinstance(genres_list[0], str) and 1 < len(lyrics_list) == len(genres_list):
            genres_list = [[g] for g in genres_list]

        assert len(lyrics_list) == len(genres_list)
        assert isinstance(genres_list[0][0], str) and isinstance(lyrics_list[0], str)
        return genres_list, lyrics_list

    def _generate_chord_line(self, model_input, chords_generating_params=None):
        if not chords_generating_params:
            chords_generating_params = CHORDS_GENERATING_PARAMS
        tokenized_input = self.tokenizer(model_input, return_tensors="pt", padding=True)
        outputs = self.chords_model.generate(
            input_ids=tokenized_input['input_ids'].to(device='cuda'),
            attention_mask=tokenized_input['attention_mask'].to(device='cuda'),
            **chords_generating_params
        )
        return self.tokenizer.batch_decode(outputs.to(device='cpu'), skip_special_tokens=True)[0]

    def write_song_chords(self, genres, lyrics, chords_generating_params=None):
        lyrics_lines = [ll for ll in lyrics.split('\n')]
        title = next(ll for ll in lyrics_lines if len(ll.replace(' ', '')) > 0)
        song_lines = [title.strip(' ')]
        prefix = 'write chords ' + '[' + ', '.join(sorted(genres)) + ']' + ':'
        current_input = [prefix]
        for lyrics_line in lyrics_lines[lyrics_lines.index(title)+1:]:
            line = lyrics_line.strip(' ')
            if len(lyrics_line.replace(' ', '')) == 0 or 'capo' in lyrics_line.lower() or \
                    '(' == line[0]:
                song_lines.append(line)
                continue
            current_input.append(line)
            chord_line = self._generate_chord_line(model_input=' @ '.join(current_input[-CHORDS_WINDOW:]),
                                                   chords_generating_params=chords_generating_params)
            current_input.insert(len(current_input) - 1, chord_line)
            song_lines += [decode_chord_line(chord_line), line]
        return '\n'.join(song_lines)

    def write_songs_chords(self, genres_list=List[List[str]], lyrics_list=List[str], lyrics_generating_params=None,
                           chords_generating_params=None):
        genres_list, lyrics_list = SongWriter.fix_write_chords_params(genres_list, lyrics_list)
        songs = []
        for genres, lyrics in zip(genres_list, lyrics_list):
            songs.append(self.write_song_chords(genres, lyrics, chords_generating_params))
        return songs

    def write_songs(self, genres_list: List[List[str]], song_inputs: List[str], lyrics_generating_params=None,
                    chords_generating_params=None, batch_size=None):
        lyrics_list = self.write_lyrics(genres_list, song_inputs,
                                        lyrics_generating_params=lyrics_generating_params, batch_size=batch_size)
        chords_and_lyrics = self.write_songs_chords(genres_list=genres_list, lyrics_list=lyrics_list,
                                                    chords_generating_params=chords_generating_params)
        return chords_and_lyrics

    
def get_best_song_writer():
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    lyrics_model = T5ForConditionalGeneration.from_pretrained(LYRICS_BEST_VERSION)
    lyrics_model = lyrics_model.to(device='cuda')
    chords_model = T5ForConditionalGeneration.from_pretrained(CHORDS_BEST_VERSION)
    chords_model = chords_model.to(device='cuda')
    song_writer = SongWriter(lyrics_model, chords_model, tokenizer, verbose=True)
    return song_writer
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

