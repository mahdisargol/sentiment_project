
from django.shortcuts import render
from .utils import predict_sentiment

def classify_sentiment(request):
    if request.method == 'POST':
        comment = request.POST['comment']
        sentiment = predict_sentiment(comment)
        return render(request, 'classify_sentiment.html', {'sentiment': sentiment})
    else:
        return render(request, 'classify_sentiment.html')
