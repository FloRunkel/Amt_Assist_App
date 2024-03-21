import tensorflow as tf

class ModelTrainer:
    def __init__(self, word_index):
        self.word_index = word_index
        self.model = None
        self.history = None

    def define_model(self):
        embedding_dim = 128
        vocab_size = len(self.word_index) + 1
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
            tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.LSTM(units=128, return_sequences=True),
            tf.keras.layers.LSTM(units=64),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=vocab_size, activation='softmax')
        ])

    def compile_and_train_model(self, train_padded, train_label_indices, test_padded, test_label_indices):
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        num_epochs = 10
        self.history = self.model.fit(train_padded, train_label_indices, epochs=num_epochs, validation_data=(test_padded, test_label_indices), verbose=1)

    def evaluate_model(self, test_padded, test_label_indices):
        test_loss, test_accuracy = self.model.evaluate(test_padded, test_label_indices, verbose=0)
        return test_loss, test_accuracy