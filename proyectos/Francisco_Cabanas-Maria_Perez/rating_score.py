import pandas as pd
# Importo el dataframe de reviews
from lectura_datasets import df_reviews

# Filtramos las columnas más relevantes

df_reviews = df_reviews[[ "asin", "reviewText", "overall" ]]

# Calculamos el promedio del overall rating por producto

product_ratings = df_reviews.groupby('asin')['overall'].mean()
