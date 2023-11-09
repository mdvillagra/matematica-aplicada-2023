import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
import json

def cargar_datos_desde_json(archivo_json):
    with open(archivo_json, 'r') as file:
        lineas = file.readlines()
        datos_json = [json.loads(line.strip()) for line in lineas if line.strip()]
    return datos_json


archivo = input("Nombre del archivo:")
datos_json = cargar_datos_desde_json(archivo)
categoria = archivo
productos = []
for dato in datos_json:
        name = dato.get('reviewerName', 'Nombre no disponible')
        rating = dato.get('overall', 0.0)
        category = categoria
        attributes = [random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)]
        weight = round(random.uniform(0, 1), 2)
        productos.append({'name':name, 'attributes':attributes,'rating': rating, 'category': category, 'weight': weight})

# Supongamos que tienes una lista de productos con atributos, calificaciones, categorías y pesos
# Cada producto es un diccionario con atributos, calificaciones, categorías y pesos
#productos = [
#    {"name": "Product A", "attributes": [1, 2, 3], "rating": 4.5, "category": "Category1", "weight": 0.2},
#    {"name": "Product B", "attributes": [2, 3, 4], "rating": 4.0, "category": "Category2", "weight": 0.3},
    # Agrega más productos aquí
#]

# Crear una lista vacía para almacenar los productos con calificaciones altas
high_rating_productos = []

# Definir un umbral de calificación
umbral_calificacion = 4.0

# Recorrer la lista de productos y seleccionar aquellos con calificaciones superiores al umbral
for product in productos:
    if product["rating"] > umbral_calificacion:
        high_rating_productos.append(product)


# Crear listas para almacenar atributos (x), calificaciones (y), categorías (C) y pesos (W)
x = []
y = []
categorias = []
pesos = []

# Recorrer los productos con calificaciones altas y extraer sus atributos, calificaciones, categorías y pesos
for product in high_rating_productos:
    atributos = product["attributes"]
    calificacion = product["rating"]
    categoria = product["category"]
    peso = product["weight"]

    # Agregar los atributos, calificaciones, categorías y pesos a las listas correspondientes
    x.append(atributos)
    y.append(calificacion)
    categorias.append(categoria)
    pesos.append(peso)


# Paso 3: Calcular la similitud del coseno entre productos
# Esto se hará para cada par de productos
similarity_matrix = cosine_similarity(x)

# Paso 4: Calcular el peso de productos no similares basado en categorías y pesos
for i in range(len(high_rating_productos)):
    for j in range(len(high_rating_productos)):
        if i != j:
            category_i = categorias[i]
            category_j = categorias[j]
            weight_i = pesos[i]
            weight_j = pesos[j]
            if category_i == category_j:
                # Mismo categoría, no hay cambios en el peso
                similarity_matrix[i][j] *= weight_i
            else:
                # Categoría diferente, aplicar una reducción basada en pesos
                similarity_matrix[i][j] *= (weight_i + weight_j) / 2

# Paso 5: Clustering (supondremos un enfoque simple basado en umbral)
umbral_similitud = 0.7  # Define un umbral de similitud
clusters = []
for i in range(len(high_rating_productos)):
    cluster = [i]
    for j in range(len(high_rating_productos)):
        if i != j and similarity_matrix[i][j] >= umbral_similitud:
            cluster.append(j)
    clusters.append(cluster)

# Paso
# Paso 5: Clustering (supondremos un enfoque simple basado en umbral)
umbral_similitud = 0.7  # Define un umbral de similitud
clusters = []
for i in range(len(high_rating_productos)):
    cluster = [i]
    for j in range(len(high_rating_productos)):
        if i != j and similarity_matrix[i][j] >= umbral_similitud:
            cluster.append(j)
    clusters.append(cluster)

# Paso 6: Eliminar productos ya comprados (simularemos que el usuario compró algunos productos)
productos_comprados = [0, 2]  # Índices de productos comprados por el usuario
recommended_productos = []
for cluster in clusters:
    for idx in cluster:
        if idx not in productos_comprados:
            recommended_productos.append(high_rating_productos[idx])

# Paso 7: Almacenar la lista de recomendaciones en una base de datos (simulado)
# En este ejemplo, solo mostraremos las recomendaciones
print("Productos recomendados:")
for product in recommended_productos:
    print(product["name"])
# Calcularemos rating score 
suma_calificaciones = 0

# Recorrer la lista de productos y sumar las calificaciones de cada producto
for product in productos:
    calificacion = product["rating"]
    suma_calificaciones += calificacion

# Ahora, la variable "suma_calificaciones" contiene la suma total de todas las calificaciones de los productos
print("Rating Score:", suma_calificaciones / len(productos))

