import pandas as pd
import gzip
import json
from sklearn.metrics.pairwise import cosine_similarity
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

# Se carga cada linea del json
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

# Se carga el dataset en un dataframe de pandas
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# Los archivos (dataset y metadata) tienen que estar en la misma carpeta que el código
# Para probar con otros datasets, cambiar el archivo que reciben "reviews" y "metadata"

# Dataframe con todas las reviews
reviews = getDF('Software_5.json.gz')

# Dataframe con la información sobre los items
metadata = getDF('meta_Software.json.gz')

# Se inicializa un diccionario para almacenar el average rating de cada producto
average_ratings = {}

# Se agrupan las reviews por 'asin' y se calcula el rating promedio
grouped_reviews = reviews.groupby('asin')
for asin, group in grouped_reviews:
    average_rating = group['overall'].mean()
    average_ratings[asin] = average_rating

# Se agrega el rating promedio al DataFrame 'metadata'
metadata['average_rating'] = metadata['asin'].map(average_ratings)

# Se cargan los items comprados por cada usuario 
user_profiles = []
for user_id, user_data in reviews.groupby('reviewerID'):
    user_profile = pd.Series(0, index=reviews['asin'].unique())
    for index, row in user_data.iterrows():
        user_profile[row['asin']] = 1
    user_profiles.append(user_profile)

# Se convierte la matriz de usuarios en un array de numpy
user_profiles_matrix = np.array(user_profiles)

# Se hallan las similitudes entre items de acuerdo a las reviews de
# cada usuario. Asi, la similitud entre el item i y el j está
# en item_similarity[i][j] y en item_similarity[j][i]
item_similarity = cosine_similarity(user_profiles_matrix)

# Se mapean los indices de los items con sus ID
item_index_to_asin = {index: asin for index, asin in enumerate(reviews['asin'].unique())}
item_asin_to_index = {asin: index for index, asin in enumerate(reviews['asin'].unique())}

# Se define el usuario para buscar las recomendaciones
user_index = 0 

# Se cargan las reviews del usuario
items_reviewed_by_user = [i for i, value in enumerate(user_profiles[user_index]) if value == 1]

# Se imprimen los items con reviews del usuario
print(f"Items reviewed by User {user_index}:")
for item_index in items_reviewed_by_user:
    item_info = metadata[metadata['asin'] == item_index_to_asin[item_index]]
    print(' *', item_info.iloc[0]['title']) 

# Se inicializa la lista de posibles recomendaciones
possible_recomendations = []

# Se buscan los items similares
for item_index in items_reviewed_by_user:   
    # Se itera por la lista de similaridades del item
    for similar_index in range(len(item_similarity[item_index])):
        # Solo se considera un item si la similitud es mayor a 0
        # También se verifica que no se esté comparando un item con si mismo
        if item_similarity[item_index][similar_index] > 0 and item_index != similar_index:
            # Se revisa que el item al que se quiere acceder esté en la lista de items
            if similar_index in item_index_to_asin:
                similar_id = item_index_to_asin [similar_index]
                # Se agrega el item si no está en la lista de posibles recomendaciones
                # y si no está en la lista de items con review del usuario
                if (similar_id not in possible_recomendations) and (similar_id not in items_reviewed_by_user) :
                    similar_info = metadata[metadata['asin'] == similar_id]
                    similar_title = similar_info.iloc[0]['title']
                    similar_score = item_similarity[item_index][similar_index]
                    similar_rating = similar_info.iloc[0]['average_rating']

                    new_item = {
                        'ID': similar_id,
                        'Title': similar_title,
                        'Similarity': similar_score,
                        'Rating': similar_rating
                    }

                    possible_recomendations.append(new_item)

    # Lo mismo para los items relacionados
    related = metadata[metadata['asin'] == item_index_to_asin[item_index]]

    for similar_id in related.iloc[0]['also_view']:
       if similar_id in item_asin_to_index:
            if (similar_id not in possible_recomendations) and (similar_id not in items_reviewed_by_user) :
                similar_index = item_asin_to_index[similar_id]
                similar_info = metadata[metadata['asin'] == similar_id]
                similar_title = similar_info.iloc[0]['title']
                similar_score = item_similarity[item_index][similar_index]
                similar_rating = similar_info.iloc[0]['average_rating']

                new_item = {
                    'ID': similar_id,
                    'Title': similar_title,
                    'Similarity': similar_score,
                    'Rating': similar_rating
                }

                possible_recomendations.append(new_item)     

    for similar_id in related.iloc[0]['also_buy']:
       if similar_id in item_asin_to_index:
            if (similar_id not in possible_recomendations) and (similar_id not in items_reviewed_by_user) :
                similar_index = item_asin_to_index[similar_id]
                similar_info = metadata[metadata['asin'] == similar_id]
                similar_title = similar_info.iloc[0]['title']
                similar_score = item_similarity[item_index][similar_index]
                similar_rating = similar_info.iloc[0]['average_rating']

                new_item = {
                    'ID': similar_id,
                    'Title': similar_title,
                    'Similarity': similar_score,
                    'Rating': similar_rating
                }

                possible_recomendations.append(new_item)  

# Se crean las variables difusas
rating = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'rating')
similitud = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'similitud')
recomendacion = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'recomendacion')

# Se defininen las funciones de membresía para las variables difusas
rating['bajo'] = fuzz.trimf(rating.universe, [0, 0, 0.5])
rating['medio'] = fuzz.trimf(rating.universe, [0, 0.5, 1])
rating['alto'] = fuzz.trimf(rating.universe, [0.5, 1, 1])

similitud['baja'] = fuzz.trimf(similitud.universe, [0, 0, 0.5])
similitud['media'] = fuzz.trimf(similitud.universe, [0, 0.5, 1])
similitud['alta'] = fuzz.trimf(similitud.universe, [0.5, 1, 1])

recomendacion['baja'] = fuzz.trimf(recomendacion.universe, [0, 0, 0.5])
recomendacion['media'] = fuzz.trimf(recomendacion.universe, [0, 0.5, 1])
recomendacion['alta'] = fuzz.trimf(recomendacion.universe, [0.5, 1, 1])

# Se definen las reglas difusas
rule1 = ctrl.Rule(rating['bajo'] & similitud['baja'], recomendacion['baja'])
rule2 = ctrl.Rule(rating['bajo'] & similitud['media'], recomendacion['baja'])
rule3 = ctrl.Rule(rating['bajo'] & similitud['alta'], recomendacion['media'])
rule4 = ctrl.Rule(rating['medio'] & similitud['baja'], recomendacion['baja'])
rule5 = ctrl.Rule(rating['medio'] & similitud['media'], recomendacion['media'])
rule6 = ctrl.Rule(rating['medio'] & similitud['alta'], recomendacion['alta'])
rule7 = ctrl.Rule(rating['alto'] & similitud['baja'], recomendacion['media'])
rule8 = ctrl.Rule(rating['alto'] & similitud['media'], recomendacion['alta'])
rule9 = ctrl.Rule(rating['alto'] & similitud['alta'], recomendacion['alta'])

# Se crea un sistema de control difuso
sistema_recomendacion = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
recomendador = ctrl.ControlSystemSimulation(sistema_recomendacion)

decision_umbral = 0.5
print(f"Recomended items based on items reviewed by User {user_index}:")
# Se recomienda cada item por logica difusa
for recomendation in possible_recomendations:
    recomendador.input['rating'] = recomendation['Rating'] 
    recomendador.input['similitud'] = recomendation['Similarity']

    # Obtener la recomendación
    recomendador.compute()
    print(recomendador.output)
    if recomendador.output['recomendacion'] >= decision_umbral:
        print('*Title: ', recomendation['Title'], ' *Average Rating: ', recomendation['Rating'], ' *Similarity: ', recomendation['Similarity'], 'Valor de recomendación:', recomendador.output['recomendacion'])
