import pathlib
import tarfile
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import joblib

def read_imdb_split(split_dir):
    split_dir = pathlib.Path(split_dir)
    texts = []
    labels = []
    for label_dir in ['pos', 'neg']:
        for text_file in (split_dir/label_dir).iterdir():
            with open(text_file, 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(0 if label_dir == 'neg' else 1)
    return texts, labels

# Read the IMDB dataset
print("Read the IMDB dataset")
X_train, y_train = read_imdb_split('aclImdb/train')
X_test, y_test = read_imdb_split('aclImdb/test')

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_sequence_length = 100  # Adjust as needed
X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=max_sequence_length))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
print('Train the model')
model.fit(X_train, y_train, batch_size=128, epochs=5)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Model accuracy:", accuracy)

# Save the model
model.save('sentiment_analysis/sentiment_model.h5')
print("Model saved as: sentiment_analysis/sentiment_model.h5")

# Save the tokenizer
tokenizer_filename = 'sentiment_analysis/tokenizer.joblib'
joblib.dump(tokenizer, tokenizer_filename)
print("Tokenizer saved as:", tokenizer_filename)
