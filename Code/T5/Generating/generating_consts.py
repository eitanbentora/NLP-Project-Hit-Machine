LYRICS_GENERATING_PARAMS = dict(num_beams=3, no_repeat_ngram_size=2, min_length=300, max_length=600, do_sample=True)
# LYRICS_GENERATING_PARAMS = dict(do_sample=True, min_length=300, max_length=400, top_k=50, top_p=0.95, temperature=0.8, no_repeat_ngram_size=3)

# CHORDS_GENERATING_PARAMS = dict(num_beams=1, no_repeat_ngram_size=2, min_length=0, max_length=100, do_sample=False)
CHORDS_GENERATING_PARAMS = dict(do_sample=True, max_length=10, top_k=50, top_p=0.95, temperature=1)

CHORDS_WINDOW = 16
LYRICS_PREFIX = 'summarise song'
CHORDS_PREFIX = 'summarise chords'


