import json
import numpy as np

# Módulo 2: Similarity Score
def similaridad_de_jaccard(conjunta_A, conjunto_B):
    """
    Calcula la similitud de Jaccard entre dos conjuntos.

    Args:
    - conjunta_A (set): Conjunto A.
    - conjunto_B (set): Conjunto B.

    Returns:
    - float: Valor de similitud de Jaccard.
    """
    interseccion = len(conjunta_A.intersection(conjunto_B))
    union = len(conjunta_A.union(conjunto_B))
    if union == 0:
        return 0
    return interseccion / union

def similitud_entre_conjuntos(producto_A, producto_B):
    """
    Calcula la similitud entre dos productos utilizando el coeficiente de Jaccard.

    Args:
    - producto_A (dict): Datos del primer producto.
    - producto_B (dict): Datos del segundo producto.

    Returns:
    - float: Valor de similitud entre los dos productos.
    """
    conjunta_A = set(producto_A.get('also_buy', []) + producto_A.get('also_view', []) + producto_A.get('category', []))
    conjunto_B = set(producto_B.get('also_buy', []) + producto_B.get('also_view', []) + producto_B.get('category', []))
    
    return similaridad_de_jaccard(conjunta_A, conjunto_B)
