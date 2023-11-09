import pandas as pd
import gzip
import json
#Truncar la tabla para no mostrar todo (reemplazar con None para que muestre completo)
pd.set_option('display.max_rows', 10)


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')
