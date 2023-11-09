import pandas as pd
import gzip
import json
import os

#---------------------------------------------------------- Lectura de datasets ----------------------------------------------------------#

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

directorio_actual = os.path.dirname(os.path.abspath(__file__))

review_archivo = 'Gift_Cards.json.gz'
ruta_completa = os.path.join(directorio_actual, review_archivo)
nombre, extension = os.path.splitext(review_archivo)

df = getDF(ruta_completa)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

metadata_archivo = 'meta_Gift_Cards.json.gz'
ruta_completa = os.path.join(directorio_actual, metadata_archivo)
nombre, extension = os.path.splitext(metadata_archivo)

df2 = getDF(ruta_completa)

#---------------------------------------------------------- Product Rating  ----------------------------------------------------------#

avg_product_rating = df.groupby(['asin'])['overall'].mean().reset_index()

# Mostrar el resultado
# print(avg_product_rating)

#---------------------------------------------------------- Similarity Score ----------------------------------------------------------#

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0  
    return intersection / union

product_sets = {}

for index, row in df2.iterrows():
    product_id = index
    also_view_set = set(row['also_view'])
    also_buy_set = set(row['also_buy'])
    asin = row['asin']
    product_sets[product_id] = (also_view_set, also_buy_set, asin)

similarity_dict = {}
for product1 in product_sets:
    asin1 = product_sets[product1][2]
    similarity_dict[asin1] = {}
    for product2 in product_sets:
        if product1 != product2:
            asin2 = product_sets[product2][2]
            set_product1 = product_sets[product1][0].union(product_sets[product1][1])
            set_product2 = product_sets[product2][0].union(product_sets[product2][1])
            similarity = jaccard_similarity(set_product1, set_product2)
            similarity_dict[asin1][asin2] = similarity

#---------------------------------------------------------- Funciones de membresia ----------------------------------------------------------#

def membership_similarity(similarity):
    poor = max(0, min((0.25 - similarity) / (0.25 - 0), 1))
    average = max(0, min((similarity - 0.25) / (0.55 - 0.25), 1, (0.8 - similarity) / (0.8 - 0.55)))
    good = max(0, min((similarity - 0.5) / (0.8 - 0.5), 1, (1 - similarity) / (1 - 0.8)))
    excellent = max(0, min((similarity - 0.7) / (1 - 0.7), 1))

    return poor, average, good, excellent

def membership_rating_score(score):
    poor = max(0, min((2.5 - score) / (2.5 - 0), 1))
    average = max(0, min((score - 2.3) / (3 - 2.3), 1, (4 - score) / (4 - 3.8)))
    good = max(0, min((score - 2.8) / (4 - 2.8), 1, (5 - score) / (5 - 4)))
    excellent = max(0, min((score - 3.8) / (5 - 3.8), 1))

    return poor, average, good, excellent

#---------------------------------------------------------- Aplicacion de reglas ----------------------------------------------------------#

def fuzzy_rules(rating_score, similarity):
    membership_score = membership_rating_score(rating_score)
    membership_sim = membership_similarity(similarity)

    rules_matrix_l = [
        ["Not Recommended", "Not Recommended", "Not Recommended", "Not Recommended"],
        ["Not Recommended", "Recommended", "Recommended", "Recommended"],
        ["Not Recommended", "Recommended", "Likely to Recommend", "Likely to Recommend"],
        ["Not Recommended", "Recommended", "Likely to Recommend", "Very Recommended"]
    ]

    rules_matrix = [
        [ 0 , 0  , 0  , 0  ],
        [ 0 , 0.3, 0.4, 0.5],
        [ 0 , 0.4, 0.8, 0.9],
        [ 0 , 0.5, 0.9, 1  ],
    ]
    
    recommendation = rules_matrix_l[membership_score.index(max(membership_score))][membership_sim.index(max(membership_sim))]
    recommendation_score = rules_matrix[membership_score.index(max(membership_score))][membership_sim.index(max(membership_sim))]

    return recommendation_score, recommendation

#---------------------------------------------------------- Función de cálculo ----------------------------------------------------------#

def find_recommendations(product_asin):
    r = {}
    for product in product_sets:
        asin = product_sets[product][2]
        if asin != product_asin:
            pr = avg_product_rating.loc[avg_product_rating['asin'] == asin]['overall'].values[0]
            sim = similarity_dict[product_asin][asin]
            recommended_score, recommendation_l = fuzzy_rules(pr,sim)
            r[product] = (asin, recommended_score, recommendation_l)

    return r

#---------------------------------------------------------- Prueba de ejecución ----------------------------------------------------------#

product_asin = 'B004LLIKVU'

result = find_recommendations(product_asin)
print('Recomendaciones para el producto: ',product_asin)

sorted_result = sorted(result.items(), key=lambda x: x[1][1], reverse=True)

for index, values in sorted_result[:10]:
    print(f"Producto: {values[0]} - {values[2]}")