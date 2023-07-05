import pathlib
import tarfile
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
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

# Feature extraction
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Build and train the model
print('Build and train the model')
model = LogisticRegression()
model.fit(X_train, y_train)
#
# # Evaluate the model
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)
#
# Save the model
model_filename = 'sentiment_analysis\sentiment_model.joblib'
joblib.dump(model, model_filename)
print("Model saved as:", model_filename)

# # Save the vectorizer
vectorizer_filename = 'sentiment_analysis\Vectorizer.joblib'
joblib.dump(vectorizer, vectorizer_filename)
print("vectorizer saved as:", vectorizer_filename)