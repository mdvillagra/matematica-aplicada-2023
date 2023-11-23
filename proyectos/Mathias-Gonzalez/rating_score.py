import pandas as pd

def calcular_rating_score(dataset):
    product_ratings = dataset.groupby("asin")[["overall"]].mean().rename(columns={"overall": "product_rating"})
    return product_ratings