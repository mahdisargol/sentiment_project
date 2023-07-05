import joblib
import os

model_filename = os.path.join(os.path.dirname(__file__), 'sentiment_model.joblib')
model = joblib.load(model_filename)

vectorizer_filename = os.path.join(os.path.dirname(__file__),'Vectorizer.joblib')
vectorizer = joblib.load(vectorizer_filename)

# Function to predict sentiment
def predict_sentiment(review):
    print(review)
    processed_review = vectorizer.transform([review])
    prediction = model.predict(processed_review)
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    return sentiment


# Test the model
review_input = "This movie was great! I really enjoyed it."


sentiment = predict_sentiment(review_input)
print("Review:", review_input)
print("Sentiment:", sentiment)