import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os 
import numpy as np
import ssl
     
from Preprocessor import Preprocessor
from model_trainer import ModelTrainer

ssl._create_default_https_context = ssl._create_unverified_context

class Chatbot:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.train_data = None
        self.test_data = None
        self.train_padded = None
        self.test_padded = None
        self.label_to_index = None
        self.model_trainer = None
        self.model_evaluation = None
        self.preprocessor = Preprocessor()

    def load_data(self):
        self.df = pd.read_csv(self.data_path, delimiter=";", header=None, names=["Question", "Answer"], encoding='latin1')

    def preprocess_and_tokenize_data(self):
        self.preprocessor.download_nltk_data()
        self.df['Processed Question'] = self.df['Question'].apply(self.preprocessor.preprocess)
        self.df['Processed Answer'] = self.df['Answer'].apply(self.preprocessor.preprocess)
        self.preprocessor.tokenize_data(self.df['Processed Question'] + self.df['Processed Answer'])

    def split_data(self):
        self.train_data, self.test_data = train_test_split(self.df, test_size=0.3, random_state=42)

    def create_padded_sequences(self):
        max_length = 20
        trunc_type = 'post'
        padding_type = 'post'
        self.train_padded = tf.keras.preprocessing.sequence.pad_sequences(self.preprocessor.tokenizer.texts_to_sequences(self.train_data['Processed Question']),
                                          maxlen=max_length, padding=padding_type, truncating=trunc_type)
        self.test_padded = tf.keras.preprocessing.sequence.pad_sequences(self.preprocessor.tokenizer.texts_to_sequences(self.test_data['Processed Question']),
                                         maxlen=max_length, padding=padding_type, truncating=trunc_type)

    def create_label_indices(self):
        self.label_to_index = {label: i for i, label in enumerate(self.df['Processed Answer'].unique())}
        self.train_data['Label Index'] = self.train_data['Processed Answer'].map(self.label_to_index)
        self.test_data['Label Index'] = self.test_data['Processed Answer'].map(self.label_to_index)

    def train_model(self):
        self.model_trainer = ModelTrainer(self.preprocessor.tokenizer.word_index)
        self.model_trainer.define_model()
        self.model_trainer.compile_and_train_model(self.train_padded, self.train_data['Label Index'],
                                                   self.test_padded, self.test_data['Label Index'])
        # Evaluierung des Modells
        self.model_evaluation = self.model_trainer.evaluate_model(self.test_padded, self.test_data['Label Index'])

    def save_model(self, model_name="trained_model"):
        # Speichern des trainierten Modells
        model_path = f"/Users/florianrunkel/Desktop/Amt_Assist_App/ai/model/{model_name}.keras"
        self.model_trainer.model.save(model_path)
    
    def evaluation(self):
        # Speichern der Evaluierungsergebnisse
        evaluation_path = "/Users/florianrunkel/Desktop/Amt_Assist_App/ai/eval/evaluation_results.txt"
        with open(evaluation_path, "w") as file:
            file.write("Evaluation Results:\n")
            file.write("Test Loss: {}\n".format(self.model_evaluation[0]))
            file.write("Test Accuracy: {}\n".format(self.model_evaluation[1]))

    def load_and_predict(self, user_input, model_name="trained_model"):
        # Laden des gespeicherten Modells
        model_path = f"/Users/florianrunkel/Desktop/Amt_Assist_App/ai/model/{model_name}.keras"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
        loaded_model = tf.keras.models.load_model(model_path)
        # loaded_model.summary() # Wenn Sie das Modell überprüfen möchten, ob es geladen wurde

        # Preprocessing der Benutzereingabe
        user_input_processed = self.preprocessor.preprocess(str(user_input))

        # Stellen Sie sicher, dass tokenizer vorhanden ist
        if self.preprocessor.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Make sure to call 'tokenize_data()' method first.")

        user_input_sequence = self.preprocessor.tokenizer.texts_to_sequences([user_input_processed])
        max_length = 20
        padding_type = 'post'
        trunc_type = 'post'
        user_input_padded = tf.keras.preprocessing.sequence.pad_sequences(user_input_sequence, maxlen=max_length,
                                        padding=padding_type, truncating=trunc_type)

        # Vorhersage treffen
        predicted_index = loaded_model.predict(user_input_padded, verbose=1)
        predicted_word_index = np.argmax(predicted_index)
        
        # Laden des Preprocessors, um auf das Vokabular zuzugreifen
        word_index = self.preprocessor.tokenizer.word_index

        print("Vorhersagter Index:", predicted_word_index)
        
        # Überprüfen des Vokabulars
        if predicted_word_index not in word_index.values():
            print("Warnung: Der vorhergesagte Index liegt nicht im Vokabular.")
            #print("Vokabular:", word_index)
            return

        # Wort basierend auf dem Index erhalten
        predicted_word = [word for word, index in word_index.items() if index == predicted_word_index][0]
        print("Vorhergesagtes Wort:", predicted_word)
        return predicted_word