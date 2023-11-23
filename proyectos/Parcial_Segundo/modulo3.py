import json
import numpy as np

def obtener_promedio_rating_por_producto(datos):
    """
    Calcula el promedio de rating por producto a partir de los datos de reviews.

    Args:
    - datos (list): Lista de objetos JSON con datos de reviews.

    Returns:
    - dict: Diccionario con el promedio de rating por producto.
    """
    rating_promedio_por_producto = {}

    for objeto_json in datos:
        asin = objeto_json['asin']
        rating = objeto_json['overall']

        if asin in rating_promedio_por_producto:
            rating_promedio_por_producto[asin]['total_rating'] += rating
            rating_promedio_por_producto[asin]['total_reviews'] += 1
        else:
            rating_promedio_por_producto[asin] = {
                'total_rating': rating,
                'total_reviews': 1
            }

    for datos_producto in rating_promedio_por_producto.values():
        datos_producto['promedio_rating'] = datos_producto['total_rating'] / datos_producto['total_reviews']

    return rating_promedio_por_producto
