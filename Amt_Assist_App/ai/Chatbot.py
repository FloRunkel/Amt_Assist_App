from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__, template_folder='templates')

class Chatbot:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.tokenizer = None
        self.word_index = None
        self.train_data = None
        self.test_data = None
        self.train_padded = None
        self.test_padded = None
        self.label_to_index = None
        self.model = None

    def download_nltk_data(self):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')

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

    def load_data(self):
        self.df = pd.read_csv(self.data_path, delimiter=";", header=None, names=["Question", "Answer"], encoding='latin1')

    def preprocess_data(self):
        self.df['Processed Question'] = self.df['Question'].apply(self.preprocess)
        self.df['Processed Answer'] = self.df['Answer'].apply(self.preprocess)

    def split_data(self):
        self.train_data, self.test_data = train_test_split(self.df, test_size=0.3, random_state=42)

    def tokenize_data(self):
        vocab_size = 10000
        oov_tok = "<OOV>"
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        self.tokenizer.fit_on_texts(self.df['Processed Question'] + self.df['Processed Answer'])
        self.word_index = self.tokenizer.word_index

    def create_padded_sequences(self):
        max_length = 20
        trunc_type = 'post'
        padding_type = 'post'
        self.train_padded = pad_sequences(self.tokenizer.texts_to_sequences(self.train_data['Processed Question']),
                                          maxlen=max_length, padding=padding_type, truncating=trunc_type)
        self.test_padded = pad_sequences(self.tokenizer.texts_to_sequences(self.test_data['Processed Question']),
                                         maxlen=max_length, padding=padding_type, truncating=trunc_type)

    def create_label_indices(self):
        self.label_to_index = {label: i for i, label in enumerate(self.df['Processed Answer'].unique())}
        self.train_label_indices = self.train_data['Processed Answer'].map(self.label_to_index)
        self.test_label_indices = self.test_data['Processed Answer'].map(self.label_to_index)

    def define_model(self):
        embedding_dim = 64
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=len(self.word_index) + 1, output_dim=embedding_dim),
            tf.keras.layers.Conv1D(64, 5, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=4),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.label_to_index), activation='softmax')
        ])

    def compile_and_train_model(self):
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        num_epochs = 10
        self.history = self.model.fit(self.train_padded, self.train_label_indices, epochs=num_epochs,
                                      validation_data=(self.test_padded, self.test_label_indices), verbose=2)

    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.test_padded, self.test_label_indices)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

    def predict(self, user_input):
        user_input_processed = self.preprocess(user_input)
        user_input_sequence = self.tokenizer.texts_to_sequences([user_input_processed])
        max_length = 20
        padding_type = 'post'
        trunc_type = 'post'
        user_input_padded = pad_sequences(user_input_sequence, maxlen=max_length,
                                          padding=padding_type, truncating=trunc_type)
        predicted_index = self.model.predict(user_input_padded, verbose=0).argmax(axis=-1)[0]
        predicted_answer = list(self.label_to_index.keys())[predicted_index]
        return predicted_answer

data_path = "/Users/florianrunkel/Desktop/Amt_Assist_App/ai/Data.csv"
chatbot = Chatbot(data_path)
chatbot.download_nltk_data()
chatbot.load_data()
chatbot.preprocess_data()
chatbot.split_data()
chatbot.tokenize_data()
chatbot.create_padded_sequences()
chatbot.create_label_indices()
chatbot.define_model()
chatbot.compile_and_train_model()

@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/chatbot.html')
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json.get('user_input')
    if user_input is None:
        return jsonify({'error': 'Missing user_input field'}), 400
    predicted_answer = chatbot.predict(user_input)
    return jsonify({'response': predicted_answer})

if __name__ == "__main__":
    app.run(debug=True)