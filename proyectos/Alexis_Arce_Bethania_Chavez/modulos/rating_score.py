import pandas as pd
from IPython.display import display, HTML
"""
Rating Score: Tomar el promedio de los ratings de cada producto
"""
def calcular_rating_score(df):
    # Calcular el rating score promedio para cada ítem (usando la columna "overall"(calificacion del producto) )
    rating_score = df.groupby('asin')['overall'].mean().reset_index()
    pd.set_option('display.max_rows', 10)
    # Mostrar el DataFrame con el rating score de cada ítem
    display(rating_score)
    return rating_score