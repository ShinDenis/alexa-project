import re
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# ── NLTK downloads ────────────────────────────────────────────────────────────
nltk.download('stopwords',                      quiet=True)
nltk.download('wordnet',                        quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('omw-1.4',                        quiet=True)

# ── Препроцессинг ───────────────────────────────────────
ps         = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

negation_words = {
    'no','nor','not',"don't","didn't","doesn't","won't","isn't","aren't",
    "wasn't","weren't","can't","cannot","couldn't","shouldn't","wouldn't",
    "haven't","hasn't","hadn't","mustn't","mightn't","shan't",
    "ain't","never","nothing","nowhere","neither","without","nobody","none"
}
stop_words_neg = stop_words - negation_words


def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN


def preprocess_stem(text):
    text  = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    words = [ps.stem(w) for w in text.split() if w not in stop_words_neg]
    return ' '.join(words)


def preprocess_lemma(text):
    text   = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    words  = [w for w in text.split() if w not in stop_words_neg]
    tagged = pos_tag(words, lang='eng')
    return ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged])


def get_device_type(variation):
    v = variation.lower()
    if 'dot'   in v: return 'Echo Dot'
    if 'show'  in v: return 'Echo Show'
    if 'spot'  in v: return 'Echo Spot'
    if 'plus'  in v: return 'Echo Plus'
    if 'stick' in v: return 'Fire TV Stick'
    return 'Echo'


def prepare_input(review_text, variation='Echo'):
    return pd.DataFrame([{
        'reviews_stemmed':    preprocess_stem(review_text),
        'reviews_lemmatized': preprocess_lemma(review_text),
        'device_type':        get_device_type(variation),
        'review_length':      len(str(review_text))
    }])