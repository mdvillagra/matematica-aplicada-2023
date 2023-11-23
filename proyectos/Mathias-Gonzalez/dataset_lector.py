import random
import pandas as pd
import gzip
import json

sample_size = 500

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path, keys, sample = True):
  n = sum(1 for d in parse(path))-1
  if(sample and n>sample_size):
    skip = sorted(random.sample(range(1, n+1), sample_size))  # selecciona numeros de registros aleatorios para mantener
    print(f'datos:{n} muestra:{sample_size}')
  else:
    print(f'datos:{n}')
    sample = False
  
  i = 0
  j = 0

  df = {}
  if sample:
    for d in parse(path):
      df[j] = {key: d[key] for key in keys if key in d.keys()}
      i += 1
      if i == skip[j]:
        j+=1
        if j == sample_size:
          break
  else:
    for d in parse(path):
      df[i] = {key: d[key] for key in keys if key in d.keys()}
      i += 1
  return pd.DataFrame.from_dict(df, orient='index')


def load_dataframe(filename: str):
  
  if filename.endswith(".csv"):
    n = sum(1 for line in open(filename))-1
    if(n>sample_size):
      skip = sorted(random.sample(range(1, n+1), n-sample_size))   # selecciona numeros de registros aleatorios para omitir
      print(f'datos:{n} muestra:{sample_size}')
      df= pd.read_csv(filename, names=["asin", "reviewerID", "overall", "unixReviewTime"], skiprows=skip)
    else:
      print(f'datos:{n}')
      df= pd.read_csv(filename, names=["asin", "reviewerID", "overall", "unixReviewTime"])
    return df
  
  elif filename.endswith(".json.gz"):
    return getDF(filename, ["asin", "reviewerID", "overall", "unixReviewTime"])
  else:
    raise TypeError("File type not supported")
  
def load_metadata(filename: str):
  if filename.endswith(".json.gz"):
    return getDF(filename, ["asin", "title", "description", "feature", "brand", "also_buy", "also_view"], False)
  else:
    raise TypeError("File type not supported")
