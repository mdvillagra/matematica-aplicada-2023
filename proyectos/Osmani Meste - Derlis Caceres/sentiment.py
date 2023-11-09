import pandas as pd
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

def parse(path):
  g = open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('Software.json')
dfmeta = getDF('meta_Software.json')
print(dfmeta.columns)

def get_sentiment_score(text):
    if not pd.isna(text):  # Comprueba si el valor no es nulo
        sentiment = sia.polarity_scores(text)
        return sentiment['compound']
    else:
        return None  # O cualquier otro valor que desees para representar un valor nulo

start = time.time()
print(start)

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

df = getDF('Software.json')

df['sentiment_score'] = df['reviewText'].apply(get_sentiment_score)

for review in df['sentiment_score'][:1000]:
    print(review)

 
end = time.time()
print(end)
print(end - start)
