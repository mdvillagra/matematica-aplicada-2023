import json
import pandas as pd
import gzip
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

#Funcion para leer los archivos que contienen los datos
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

#Funcion para convertir los datos en dataframes
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('meta_Appliances.json.gz')

#Funcion para encontrar similaridad entre productos
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection

    return intersection / union

producto_especifico_asin = df['asin'][6]
producto_especifico_compras = set(df[df['asin'] == producto_especifico_asin]['also_buy'].values[0])
producto_especifico_historial = set(df[df['asin'] == producto_especifico_asin]['also_view'].values[0])

similaridades = []

for index, row in df.iterrows():
    if row['asin'] != producto_especifico_asin:
        asin = row['asin']
        also_buy = set(row['also_buy'])
        also_viewed = set(row['also_view'])
        compras_similarity = jaccard_similarity(producto_especifico_compras, also_buy)
        historial_similarity = jaccard_similarity(producto_especifico_historial, also_viewed)
        similarity_score = (compras_similarity + historial_similarity) / 2
        similaridades.append({'asin': asin, 'similarity_score': similarity_score})

def funcion_membresia_triangular(valor, a, b, c):
    if a < valor < b:
        return (valor - a) / (b - a)
    elif b <= valor < c:
        return (c - valor) / (c - b)
    else:
        return 0.0
    
# Encontrar los valores mínimo y máximo de similitud en la lista de similaridades
valores_similitud = [producto['similarity_score'] for producto in similaridades]
valor_minimo = min(valores_similitud)
valor_maximo = max(valores_similitud)

a = valor_minimo
c = valor_maximo
b = (a + c) / 2

# Calcular la membresía para cada producto
for producto in similaridades:
    similarity_score = producto['similarity_score']
    membresia = funcion_membresia_triangular(similarity_score, a, b, c)
    producto['membresia'] = membresia

df_similaridades = pd.DataFrame(similaridades)

ruta_archivo = 'similarity_scores.csv'

df_similaridades.to_csv(ruta_archivo, index=False)

df1 = getDF('Appliances_5.json.gz')

aggregated_df1 = df1.groupby('asin')['overall'].agg(['sum', 'count']).reset_index()

aggregated_df1.columns = ['asin', 'suma_overall', 'conteo_overall']

# Calcular el promedio
aggregated_df1['rating_score'] = aggregated_df1['suma_overall'] / aggregated_df1['conteo_overall']

# Encontrar los valores mínimo y máximo de los puntajes de valoración
valores_rating = aggregated_df1['rating_score']
valor_minimo = valores_rating.min()
valor_maximo = valores_rating.max()

a = valor_minimo
c = valor_maximo
b = (a + c) / 2  

# Calcular la membresía para cada producto
for index, row in aggregated_df1.iterrows():
    rating_score = row['rating_score']
    membresia = funcion_membresia_triangular(rating_score, a, b, c)
    aggregated_df1.at[index, 'membresia'] = membresia

ruta_archivo = 'rating_scores.csv'

aggregated_df1.to_csv(ruta_archivo, index=False)

interseccion = pd.merge(df_similaridades, aggregated_df1, on='asin', how='inner')

# Definir las etiquetas de membresía para las variables de entrada
similarity_score = ctrl.Antecedent(universe=[0, 1], label='Similarity Score')
rating_score = ctrl.Antecedent(universe=[1, 5], label='Rating Score')

similarity_score.automf(3, names=['Baja', 'Media', 'Alta'])
rating_score.automf(3, names=['Baja', 'Media', 'Alta'])

# Definir las etiquetas de membresía para las variables de salida
recomendacion = ctrl.Consequent(universe=[0, 1], label='Recomendacion')
recomendacion.automf(4, names=['No Recomendado', 'Medianamente Recomendado', 'Recomendado', 'Altamente Recomendado'])

# Definir reglas difusas
rule1 = ctrl.Rule((similarity_score['Alta'] | rating_score['Alta']), recomendacion['Altamente Recomendado'])
rule2 = ctrl.Rule((similarity_score['Alta'] | rating_score['Media']), recomendacion['Recomendado'])
rule3 = ctrl.Rule((similarity_score['Media'] | rating_score['Alta']), recomendacion['Recomendado'])
rule4 = ctrl.Rule((similarity_score['Media'] | rating_score['Media']), recomendacion['Medianamente Recomendado'])
rule5 = ctrl.Rule((similarity_score['Media'] | rating_score['Baja']), recomendacion['No Recomendado'])
rule6 = ctrl.Rule((similarity_score['Baja'] | rating_score['Media']), recomendacion['No Recomendado'])
rule7 = ctrl.Rule((similarity_score['Baja'] | rating_score['Baja']), recomendacion['No Recomendado'])
rule8 = ctrl.Rule((similarity_score['Alta'] | rating_score['Baja']), recomendacion['Medianamente Recomendado'])
rule9 = ctrl.Rule((similarity_score['Baja'] | rating_score['Alta']), recomendacion['Medianamente Recomendado'])

# Crear el sistema de control difuso
sistema_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

simulador = ctrl.ControlSystemSimulation(sistema_control)

# Obtener listas de valores de similarity_score y rating_score para todos los casos
valores_similarity_score = interseccion['similarity_score'].values
valores_rating_score = interseccion['rating_score'].values

recomendaciones_finales = []

# Iterar sobre los casos y calcular las recomendaciones para cada uno
for i in range(len(valores_similarity_score)):
    simulador.input['Similarity Score'] = valores_similarity_score[i]
    simulador.input['Rating Score'] = valores_rating_score[i]
    simulador.compute()
    recomendacion_final = simulador.output['Recomendacion']
    recomendaciones_finales.append(recomendacion_final)
    interseccion.at[i, 'Recomendacion'] = recomendacion_final

# Imprimir las recomendaciones para cada caso
print(interseccion)

#similarity_score.view()
#rating_score.view()
#recomendacion.view()
#plt.show()
