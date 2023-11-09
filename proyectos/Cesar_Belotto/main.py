#importe de librerias a ser utilizadas

import pandas as pd
import numpy as np
import gzip
import json



"""
MODULO 1
Lectura de DataSets

Procedemos a leer los datasets de Software_5.json.gz guardandolo en la variable ratings y
el dataset de metadatos
"""

print("Modulo en progreso:\n MODULO 1: Lectura de Datasets")

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  ratings = {}
  for d in parse(path):
    ratings[i] = d
    i += 1
  return pd.DataFrame.from_dict(ratings, orient='index')

ratings = getDF('Software.json.gz')
#ratings = getDF('Datos_validos\Gift_Cards.json.gz')


# Lectura de metadatos correspondiente a Metadata
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

meta = getDF('meta_Software.json.gz')
#meta = getDF('Datos_validos\meta_Gift_Cards.json.gz')



"""
MODULO 2
Calculo del Similarity Score
"""
print("Modulo en progreso:\n MODULO 2: Calculo del Similarity Score")


def similaridad_con_jaccard(dataset, asin1, asin2):
    """
    Calcula la similitud de Jaccard entre dos productos en un conjunto de datos.

    Parametros:
    dataset (DataFrame): El conjunto de datos que contiene la información de los productos.
    asin1 (str): El identificador del primer producto (ASIN).
    asin2 (str): El identificador del segundo producto (ASIN).

    Retorna:
    float: El valor de similitud de Jaccard entre los 'also_view' y 'also_buy' de los dos productos.
    """
    # Obtener los 'also_view' y 'also_buy' de los dos ASIN
    data_asin1 = dataset[dataset['asin'] == asin1]
    data_asin2 = dataset[dataset['asin'] == asin2]

    set_also_view_asin1 = set(data_asin1['also_view'].iloc[0]) if len(data_asin1['also_view']) > 0 else set()
    set_also_view_asin2 = set(data_asin2['also_view'].iloc[0]) if len(data_asin2['also_view']) > 0 else set()

    set_also_buy_asin1 = set(data_asin1['also_buy'].iloc[0]) if len(data_asin1['also_buy']) > 0 else set()
    set_also_buy_asin2 = set(data_asin2['also_buy'].iloc[0]) if len(data_asin2['also_buy']) > 0 else set()

    # Combinar 'also_view' y 'also_buy' para cada ASIN
    view_mas_buy_set_asin1 = set_also_view_asin1.union(set_also_buy_asin1)
    view_mas_buy_set_asin2 = set_also_view_asin2.union(set_also_buy_asin2)

    # Calcular la similaridad de Jaccard para la combinación
    jaccard_view_mas_buy = len(view_mas_buy_set_asin1.intersection(view_mas_buy_set_asin2)) / len(view_mas_buy_set_asin1.union(view_mas_buy_set_asin2)) if len(view_mas_buy_set_asin1.union(view_mas_buy_set_asin2)) > 0 else 0

    return jaccard_view_mas_buy

#Utilizando de ejemplo el producto con ASIN = 0077500741

asin = '0077500741'
#asin = 'B004Q7CK9M'
similaritys = []

for index, row in meta.iterrows():
  asin_valor = row['asin']
  similarity = similaridad_con_jaccard(meta, asin, asin_valor)
  similaritys.append(similarity)



"""
MODULO 3

Calculo del rating Score

"""

print("Modulo en progreso:\n MODULO 3: Calculo del rating Score")

# Calcular el Rating_Score de "overall" para cada "asin"
rating_score = ratings.groupby("asin")["overall"].mean()

"""
MODULO 4

Fusificacion, defusificación, recomendación

"""

print("Modulo en progreso:\n MODULO 4: Fusificacion, defusificación, recomendación")

# Funciones de membresía triangular para Rating Score
def rating_membresia_pobre(x):
    if x <= 0 or x >= 2.5:
        return 0
    elif 0 < x <= 1.25:
        return x / 1.25
    else:
        return (2.5 - x) / 1.25

def rating_membresia_promedio(x):
    if 2.3 <= x <= 3:
        return min((x - 2.3) / (3 - 2.3), (3 - x) / (3 - 2.5))
    else:
        return 0

def rating_membresia_bueno(x):
    if 2.8 <= x <= 4:
        return min((x - 2.8) / (4 - 2.8), (4 - x) / (4 - 3.8))
    else:
        return 0

def rating_membresia_excelente(x):
    if 3.8 <= x <= 5:
        return min((x - 3.8) / (5 - 3.8), 1)
    else:
        return 0
    
# Definir las funciones de membresía para Similiraty Score
def similarity_membresia_pobre(x):
    if x <= 0.3:
        if x <= 0:
            return 1
        elif 0 < x <= 0.15:
            return 1 - (x / 0.15)
        elif 0.15 < x <= 0.3:
            return 0
    else:
        return 0

def similarity_membresia_promedio(x):
    if 0.25 <= x <= 0.4:
        return (x - 0.25) / (0.4 - 0.25)
    elif 0.4 < x <= 0.55:
        return (0.55 - x) / (0.55 - 0.4)
    else:
        return 0

def similarity_membresia_bueno(x):
    if 0.5 <= x <= 0.65:
        return (x - 0.5) / (0.65 - 0.5)
    elif 0.65 < x <= 0.8:
        return (0.8 - x) / (0.8 - 0.65)
    else:
        return 0

def similarity_membresia_excelente(x):
    if 0.7 <= x:
        if x <= 1:
            return (x - 0.7) / (1 - 0.7)
        else:
            return 1
    else:
        return 0
    
# Definir las funciones de membresía para el consecuente
def consecuente_membresia_No_Recomendado(x):
    if x <= 0.3:
        if x <= 0:
            return 1
        elif 0 < x <= 0.15:
            return 1 - (x / 0.15)
        elif 0.15 < x <= 0.3:
            return 0
    else:
        return 0

def consecuente_membresia_Probablemente_Recomendado(x):
    if 0.25 <= x <= 0.4:
        return (x - 0.25) / (0.4 - 0.25)
    elif 0.4 < x <= 0.55:
        return (0.55 - x) / (0.55 - 0.4)
    else:
        return 0

def consecuente_membresia_Recomendado(x):
    if 0.5 <= x <= 0.65:
        return (x - 0.5) / (0.65 - 0.5)
    elif 0.65 < x <= 0.8:
        return (0.8 - x) / (0.8 - 0.65)
    else:
        return 0

def consecuente_membresia_Altamente_Recomendado(x):
    if 0.7 <= x:
        if x <= 1:
            return (x - 0.7) / (1 - 0.7)
        else:
            return 1
    else:
        return 0
    
#Algoritmo de Recomendacion

def firing_strengths(similiraty_score,rating_score,antecedente_similarity, antecedente_rating):

  if antecedente_similarity == 0:
    firing_strengths_similarity = similarity_membresia_pobre(similiraty_score)
    
  elif antecedente_similarity == 1:
    firing_strengths_similarity = similarity_membresia_promedio(similiraty_score)
    
  elif antecedente_similarity == 2:
    firing_strengths_similarity = similarity_membresia_bueno(similiraty_score)
    
  else:
    firing_strengths_similarity = similarity_membresia_excelente(similiraty_score)
    

  if antecedente_rating == 0:
    firing_strengths_rating = rating_membresia_pobre(rating_score)
    
  elif antecedente_rating == 1:
    firing_strengths_rating = rating_membresia_promedio(rating_score)
   
  elif antecedente_rating == 2:
    firing_strengths_rating = rating_membresia_bueno(rating_score)
    
  else:
    firing_strengths_rating = rating_membresia_excelente(rating_score)
   

  #Combinamos las fuerzas de activacion

  combinado_fired_strengths = min(firing_strengths_similarity, firing_strengths_rating)

  return combinado_fired_strengths  

def calculo_inferencia(consecuente_recomendacion, firing_strength):

  #Utilizamos para el calculo la inferencia de Mamdani

  y_valores = np.linspace(0,1,10)


  b_prima = []

  if consecuente_recomendacion == 0:
    for y in y_valores:
      min_values = min(firing_strength, consecuente_membresia_No_Recomendado(y))
      b_prima.append(min_values)
  elif consecuente_recomendacion == 1:
    for y in y_valores:
      min_values = min(firing_strength, consecuente_membresia_Probablemente_Recomendado(y))
      b_prima.append(min_values)
  elif consecuente_recomendacion == 2:
    for y in y_valores:
      min_values = min(firing_strength, consecuente_membresia_Recomendado(y))
      b_prima.append(min_values)
  else:
    for y in y_valores:
      min_values = min(firing_strength, consecuente_membresia_Altamente_Recomendado(y))
      b_prima.append(min_values)
  
  return b_prima

def defuzificar(b_prima, consecuente_recomendacion):

  y_valores = np.linspace(0,1,10)
  numerador = 0

  
  for y, b in zip(y_valores, b_prima):
      numerador += y * b


  """
  for i in range(len(b_prima)):
    numerador += y_valores[i] * b_prima[i]
  """

  denominador = sum(b_prima)

  if denominador != 0:
    return numerador/denominador
  else:
    return 0


def obtener_recomendacion(similiraty_score, rating_score,antecedente_similarity, antecedente_rating, consecuente_recomendacion):

  firing_strength = firing_strengths(similiraty_score,rating_score,antecedente_similarity, antecedente_rating)

  b_prima = calculo_inferencia(consecuente_recomendacion, firing_strength)

  y_0 = defuzificar(b_prima, consecuente_recomendacion)

  return y_0


"""
Fuzzy Rules
1- SI similarity == excelente && Rating == excelente, ENTONCES Recomendacion = Altamente_Recomendado
2- SI similarity == excelente && Rating == bueno, ENTONCES Recomendacion = Altamente_Recomendado
3- SI similarity == excelente && Rating == promedio, ENTONCES Recomendacion = Altamente_Recomendado
4- SI similarity == bueno && Rating == excelente, ENTONCES Recomendacion = Altamente_Recomendado
5- SI similarity == bueno && Rating == bueno, ENTONCES Recomendacion = Recomendado
6- SI similarity == bueno && Rating == promedio, ENTONCES Recomendacion = Recomendado
7- SI similarity == promedio && Rating == excelente, ENTONCES Recomendacion = Recomendado
8- SI similarity == pobre && Rating == excelente, ENTONCES Recomendacion = Recomendado
9- SI similarity == excelente && Rating == pobre, ENTONCES Recomendacion = Probablemente_Recomendado
10- SI similarity == bueno && Rating == pobre, ENTONCES Recomendacion = Probablemente_Recomendado
11- SI similarity == promedio && Rating == bueno, ENTONCES Recomendacion = Probablemente_Recomendado
12- SI similarity == promedio && Rating == promedio, ENTONCES Recomendacion = Probablemente_Recomendado
13- SI similarity == promedio && Rating == pobre, ENTONCES Recomendacion = No_Recomendado
14- SI similarity == pobre && Rating == bueno, ENTONCES Recomendacion = No_Recomendado
15- SI similarity == pobre && Rating == promedio, ENTONCES Recomendacion = No_Recomendado
16- SI similarity == pobre && Rating == pobre, ENTONCES Recomendacion = No_Recomendado
"""

asins = meta['asin'].to_list()
lista_evaluar = []


for i in range(len(asins)):
  r = []

  if(asins[i] in rating_score.index):
    r.append(asins[i])
    r.append(similaritys[i])
    r.append(rating_score[asins[i]])
    lista_evaluar.append(r)



recomendacion = []

for l in lista_evaluar:
  """
  obtener_recomendacion(similiraty_score, rating_score,antecedente_similarity, antecedente_rating, consecuente_recomendacion)
  0 - pobre / no recomendado
  1 - promedio / probablemente_recomendado
  2 - bueno / recomendado
  3 - excelente / altamente recomendado
  """
  #Regla 1
  y_0 = obtener_recomendacion(l[1], l[2],3,3,3)
  altamente_recomendado = max(y_0,0)

  #Regla 2
  y_0 = obtener_recomendacion(l[1], l[2],3,2,3)
  altamente_recomendado = max(y_0,altamente_recomendado)

  #Regla 3
  y_0 = obtener_recomendacion(l[1], l[2],3,1,3)
  altamente_recomendado = max(y_0,altamente_recomendado)

  #Regla 4
  y_0 = obtener_recomendacion(l[1], l[2],2,3,3)
  altamente_recomendado = max(y_0,altamente_recomendado)

  #Regla 5
  y_0 = obtener_recomendacion(l[1], l[2],2,2,2)
  recomendado = max(y_0,0)

  #Regla 6
  y_0 = obtener_recomendacion(l[1], l[2],2,1,2)
  recomendado = max(y_0,recomendado)

  #Regla 7
  y_0 = obtener_recomendacion(l[1], l[2],1,3,2)
  recomendado = max(y_0,recomendado)

  #Regla 8
  y_0 = obtener_recomendacion(l[1], l[2],0,3,2)
  recomendado = max(y_0,recomendado)

  #Regla 9
  y_0 = obtener_recomendacion(l[1], l[2],3,0,1)
  probablemente_recomendado = max(y_0,0)

  #Regla 10
  y_0 = obtener_recomendacion(l[1], l[2],2,0,1)
  probablemente_recomendado = max(y_0,probablemente_recomendado)
  
  #Regla 11
  y_0 = obtener_recomendacion(l[1], l[2],1,2,1)
  probablemente_recomendado = max(y_0,probablemente_recomendado)

  #Regla 12
  y_0 = obtener_recomendacion(l[1], l[2],1,1,1)
  probablemente_recomendado = max(y_0,probablemente_recomendado)

  #Regla 13
  y_0 = obtener_recomendacion(l[1], l[2],1,0,0)
  no_recomendado = max(y_0,0)

  #Regla 14
  y_0 = obtener_recomendacion(l[1], l[2],0,2,0)
  no_recomendado = max(y_0,no_recomendado)

  #Regla 15
  y_0 = obtener_recomendacion(l[1], l[2],0,1,0)
  no_recomendado = max(y_0,no_recomendado)

  #Regla 16
  y_0 = obtener_recomendacion(l[1], l[2],0,0,0)
  no_recomendado = max(y_0,no_recomendado)

  mayor = max(altamente_recomendado, recomendado, probablemente_recomendado, no_recomendado)


  if ( altamente_recomendado == mayor):
    recomendacion.append(3)
  elif ( recomendado == mayor):
    recomendacion.append(2)
  elif ( probablemente_recomendado == mayor):
    recomendacion.append(1)
  else:
    recomendacion.append(0)

i = 0
for l in lista_evaluar:
  
  l.append(recomendacion[i])
  i = i+1



asins = meta['asin'].to_list()
lista_recomendaciones = []

for i in range(len(recomendacion)):
  r = []

  r.append(asins[i])
  r.append(recomendacion[i])

  lista_recomendaciones.append(r)


#Ordenamos la fila de forma descendente
lista_recomendaciones_ordenada_desc = sorted(lista_evaluar, key=lambda x: x[3], reverse=True)



# Imprimir top 10 de recomendaciones

titulo = meta.loc[meta['asin'] == asin, 'title'].values
print("Producto tenido en cuenta: ")
print(f'Asin: {asin}, Producto: {titulo}')

print("TOP 10 de Recomendaciones")
for i in range(11):
  if lista_recomendaciones_ordenada_desc[i][3] == 0:
    r = "No recomendado"
  elif lista_recomendaciones_ordenada_desc[i][3] == 1:
    r = "Probablemente recomendado"
  elif lista_recomendaciones_ordenada_desc[i][3] == 2:
    r = "Recomendado"
  else:
    r = "Altamente Recomendado"

  titulo = meta.loc[meta['asin'] == lista_recomendaciones_ordenada_desc[i][0], 'title'].values
  
  if(asin != lista_recomendaciones_ordenada_desc[i][0]):
    print(f'{i}- Asin: {lista_recomendaciones_ordenada_desc[i][0]}, Producto: {titulo},  {r}')

