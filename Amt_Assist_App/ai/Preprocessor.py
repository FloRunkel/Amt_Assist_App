import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import tensorflow as tf

import string

class Preprocessor:
    def __init__(self):
        self.tokenizer = None
        self.word_index = None

    def download_nltk_data(self):
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)

    def preprocess(self, data):
        if isinstance(data, str):
            tokens = nltk.word_tokenize(data.lower())
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            return " ".join(tokens)
        else:
            return ""

    def tokenize_data(self, data):
        vocab_size = 10000
        oov_tok = "<OOV>"
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        self.tokenizer.fit_on_texts(data)
        self.word_index = self.tokenizer.word_index
