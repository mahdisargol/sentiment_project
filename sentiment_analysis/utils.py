import joblib
import os

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'sentiment_model.joblib')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'Vectorizer.joblib')

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_sentiment(review):
    """
    Predicts the sentiment (positive or negative) of a given review.

    Args:
        review (str): The input review.

    Returns:
        str: The predicted sentiment.
    """
    processed_review = vectorizer.transform([review])
    prediction = model.predict(processed_review)
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    return sentiment


# Example usage
if __name__ == '__main__':
    review_input = "This movie was great! I really enjoyed it."
    sentiment = predict_sentiment(review_input)
    print("Review:", review_input)
    print("Sentiment:", sentiment)
