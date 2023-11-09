
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json

from nltk.tag import pos_tag
import nltk
import string

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Crear las variables de control y las funciones de pertenencia
rating = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'rating')
similarity = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'similarity')
recommendation = ctrl.Consequent(np.arange(0, 11, 1), 'recommendation')

# Definir las funciones de membresía con los rangos correctos
rating['poor'] = fuzz.trapmf(rating.universe, [0, 0, 2.3, 2.5])
rating['average'] = fuzz.trimf(rating.universe, [2.3, 3, 4])
rating['good'] = fuzz.trimf(rating.universe, [3, 3.8, 5])
rating['excellent'] = fuzz.trapmf(rating.universe, [3.8, 5, 5, 5])


similarity['poor'] = fuzz.trapmf(similarity.universe, [0, 0, 2.5, 3])
similarity['average'] = fuzz.trimf(similarity.universe, [2.5, 5.5, 7])
similarity['good'] = fuzz.trimf(similarity.universe, [5, 7.5, 8])
similarity['excellent'] = fuzz.trapmf(similarity.universe, [7, 10, 10, 10])


# Definir las categorías de recomendación con un rango de 0 a 10
recommendation['not_recommended'] = fuzz.trimf(recommendation.universe, [0, 0, 3])
recommendation['likely_recommended'] = fuzz.trimf(recommendation.universe, [2, 5, 7])
recommendation['recommended'] = fuzz.trimf(recommendation.universe, [6, 8, 9])
recommendation['highly_recommended'] = fuzz.trapmf(recommendation.universe, [8, 10, 10, 10])

# Reglas con 'excellent' rating
rule1 = ctrl.Rule(rating['excellent'] & similarity['excellent'], recommendation['highly_recommended'])
rule2 = ctrl.Rule(rating['excellent'] & similarity['good'], recommendation['highly_recommended'])
rule3 = ctrl.Rule(rating['excellent'] & similarity['average'], recommendation['recommended'])
rule4 = ctrl.Rule(rating['excellent'] & similarity['poor'], recommendation['likely_recommended'])

# Reglas con 'good' rating
rule5 = ctrl.Rule(rating['good'] & similarity['excellent'], recommendation['highly_recommended'])
rule6 = ctrl.Rule(rating['good'] & similarity['good'], recommendation['recommended'])
rule7 = ctrl.Rule(rating['good'] & similarity['average'], recommendation['likely_recommended'])
rule8 = ctrl.Rule(rating['good'] & similarity['poor'], recommendation['not_recommended'])

# Reglas con 'average' rating
rule9 = ctrl.Rule(rating['average'] & similarity['excellent'], recommendation['recommended'])
rule10 = ctrl.Rule(rating['average'] & similarity['good'], recommendation['likely_recommended'])
rule11 = ctrl.Rule(rating['average'] & similarity['average'], recommendation['likely_recommended'])
rule12 = ctrl.Rule(rating['average'] & similarity['poor'], recommendation['not_recommended'])

# Reglas con 'poor' rating
rule13 = ctrl.Rule(rating['poor'] & similarity['excellent'], recommendation['likely_recommended'])
rule14 = ctrl.Rule(rating['poor'] & similarity['good'], recommendation['not_recommended'])
rule15 = ctrl.Rule(rating['poor'] & similarity['average'], recommendation['not_recommended'])
rule16 = ctrl.Rule(rating['poor'] & similarity['poor'], recommendation['not_recommended'])




# Crear y simular el sistema de control
recommendation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4 , rule5, rule6, rule7, rule8, rule9, rule10, rule11,rule12, rule13, rule14, rule15, rule16])
recommendation_system = ctrl.ControlSystemSimulation(recommendation_ctrl)

# Pasar entradas al sistema de ControlSystem
def determine_recommendation_level_fuzzy(rating_score, similarity_score):


    # Asignar las entradas al sistema de control
    recommendation_system.input['rating'] = rating_score
    recommendation_system.input['similarity'] = similarity_score * 10.0

    # Calcular el resultado
    recommendation_system.compute()

    return recommendation_system.output['recommendation']

def classify_recommendation(score):
    # Define el universo de discurso para las puntuaciones de recomendación
    x_recommendation = np.arange(0, 11, 1)

    # Define los conjuntos difusos
    not_recommended = fuzz.trimf(x_recommendation, [0, 0, 3])
    likely_recommended = fuzz.trimf(x_recommendation, [2, 5, 7])
    recommended = fuzz.trimf(x_recommendation, [6, 8, 9])
    highly_recommended = fuzz.trapmf(x_recommendation, [8, 10, 10, 10])

    # Calcula el grado de pertenencia de la puntuación en cada categoría
    not_recommended_level = fuzz.interp_membership(x_recommendation, not_recommended, score)
    likely_recommended_level = fuzz.interp_membership(x_recommendation, likely_recommended, score)
    recommended_level = fuzz.interp_membership(x_recommendation, recommended, score)
    highly_recommended_level = fuzz.interp_membership(x_recommendation, highly_recommended, score)

    # Determina la categoría dominante
    levels = {
        "Not Recommended": not_recommended_level,
        "Likely Recommended": likely_recommended_level,
        "Recommended": recommended_level,
        "Highly Recommended": highly_recommended_level
    }

    max_category = max(levels, key=levels.get)
    return max_category, levels[max_category]



nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


rating_categories = [
    (0, 2.5, 'Poor'),
    (2.3, 3, 'Average'),
    (2.8, 4, 'Good'),
    (3.8, 5, 'Excellent')
    ]

# Definición de los límites de las categorías para similitudes
similarity_categories = [
    (0, 0.3, 'Poor'),
    (0.25, 0.55, 'Average'),
    (0.5, 0.8, 'Good'),
    (0.7, 1, 'Excellent')  # Asumo que excellent_lower debería ser 0.7 en lugar de -0.7
    ]

## Modulo 1 
def load_and_preprocess_reviews(file_path):
    # Cargar los datos
    data = pd.read_json(file_path, lines=True)
    
    # Limpieza básica de datos
    data['reviewText'] = data['reviewText'].fillna('')
    data['summary'] = data['summary'].fillna('')
    
    # Combinar 'reviewText' y 'summary' para obtener una representación completa de la reseña
    data['full_review'] = data['reviewText'] + " " + data['summary']
    
    # Convertir todo a minúsculas
    data['full_review'] = data['full_review'].str.lower()
    
    # Eliminar duplicados
    data = data.drop_duplicates(subset=['full_review', 'unixReviewTime'], keep='first')
    
    
    # Eliminación de puntuación y stopwords
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text):
        # Tokenizar el texto
        words = word_tokenize(text)
        # Eliminar puntuación y stopwords y lematizar
        words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
        return " ".join(words)
    
    # Aplicar la limpieza al texto completo de la reseña
    data['full_review'] = data['full_review'].apply(clean_text)

    data.reset_index(drop=True, inplace=True)

    
    return data


### modulo 2 

def calculate_similarity_between_products(grouped_reviews, tfidf_vectorizer):
    """
    Calcula la similitud entre productos basada en sus reseñas agrupadas.

    Parameters:
    - grouped_reviews: Un Series de pandas donde el índice es el 'asin' y el valor es la lista de reseñas.
    - tfidf_vectorizer: Un vectorizador TF-IDF ya ajustado.

    Returns:
    - Una matriz de similitud de coseno entre los productos.
    """

    # Unir todas las reseñas de cada producto en una única cadena de texto.
    concatenated_reviews = grouped_reviews.apply(lambda reviews: ' '.join(reviews))

    # Vectorizar las reseñas concatenadas para cada producto.
    tfidf_matrix = tfidf_vectorizer.transform(concatenated_reviews)

    # Calcular la similitud del coseno entre los vectores TF-IDF de los productos.
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)

    return cosine_sim_matrix

def find_similar_products(product_id, data, similarity_matrix):
    """
    Finds the 10 most similar products based on cosine similarity of reviews.
    
    Parameters:
    - product_id: The ASIN (unique identifier) of the product.
    - data: The preprocessed reviews dataframe.
    - similarity_matrix: The cosine similarity matrix of the reviews.
    
    Returns:
    - A list of tuples containing the ASINs and similarity scores of the top 10 similar products.
    """
    # Get the index of the product that matches the product_id
    product_idx = data.index[data['asin'] == product_id].tolist()[0]
    
    # Get the pairwise similarity scores of all products with that product
    # and convert it to a list of tuples as (position, similarity_score)
    similarity_scores = list(enumerate(similarity_matrix[product_idx]))
    
    # Sort the products based on the similarity scores in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar products, excluding the first one as it is the product itself
    most_similar_products = similarity_scores[1:11]
    
    # Get the product IDs (ASINs) of the most similar products
    similar_product_ids = [(data.iloc[i[0]]['asin'], i[1]) for i in most_similar_products]
    
    return similar_product_ids

# Modulo 3
def calculate_rating_score(data):
    data['rating_score'] = data.groupby('asin')['overall'].transform('mean')
    return data


# modulo 4??
def determine_category(score, categories):
    """
    Determines the category of a product or similarity score based on provided boundaries.
    
    Parameters:
    - score: A float representing the score of the product or similarity.
    - categories: A list of tuples where each tuple contains the lower bound, upper bound, and category name.
    
    Returns:
    - A string indicating the category of the score.
    """
    for lower, upper, name in categories:
        if lower <= score <= upper:
            # Si está más cerca del límite superior, se considera la categoría superior, de lo contrario la inferior
            return name if score > (lower + upper) / 2 else categories[categories.index((lower, upper, name)) - 1][2]

    # Si la puntuación excede el límite superior de la última categoría, aún se considera esa categoría
    return categories[-1][2]



def get_recommendations_for_all_products(data, similarity_matrix, top_n=10):
    product_recommendations = []
    asin_list = data['asin'].unique()
    asin_to_index = {asin: i for i, asin in enumerate(data['asin'])}

    for asin in asin_list:
        product_idx = asin_to_index[asin]
        similarity_scores = list(enumerate(similarity_matrix[product_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        most_similar_products = similarity_scores[1:top_n+1]  # +1 porque excluimos el producto en sí
        
        recommendations = []
        for i, score in most_similar_products:
            similar_asin = data.iloc[i]['asin']
            recommendation = determine_recommendation_level_fuzzy(float(data.iloc[i]['rating_score']), float(score))
            recommendations.append({'product-r': similar_asin, 'status': recommendation})
        
        product_recommendations.append({'product-n': asin, 'recommendations': recommendations})
    
    return product_recommendations


def count_distinct_words(file_path):
    # Load data from a JSON file
    data = pd.read_json(file_path, lines=True)
    
    # Basic data cleaning
    data['reviewText'] = data['reviewText'].fillna('')
    data['summary'] = data['summary'].fillna('')
    
    # Combine 'reviewText' and 'summary' to get a full representation of the review
    data['full_review'] = data['reviewText'] + " " + data['summary']
    
    # Remove duplicates based on 'full_review' and 'unixReviewTime'
    data = data.drop_duplicates(subset=['full_review', 'unixReviewTime'], keep='first')

    # Calculate the average of the ratings
    data['rating_score'] = data.groupby('asin')['overall'].transform('mean')
    
    #get the distinct words
    distinct_words = set()
    for review in data['full_review']:
        distinct_words.update(review.split())
    return len(distinct_words)

def words_more_than_10(file_path):
    data = pd.read_json(file_path, lines=True)
    
    data['full_review'] = (data['reviewText'].fillna('') + " " + data['summary'].fillna('')).str.lower()
    
    # Tokeniza las reseñas y etiqueta cada token
    tokens_pos = [pos_tag(word_tokenize(review)) for review in data['full_review']]
    
    # Filtra los nombres propios y las stopwords
    filtered_words = [
        word for review in tokens_pos for word, pos in review
        if pos not in ['NNP', 'NNPS'] and word not in stopwords.words('english') and word not in string.punctuation
    ]
    
    # Cuenta la frecuencia de cada palabra
    word_freq = Counter(filtered_words)
    
    # Filtra las palabras que aparecen más de 10 veces
    common_words = {word: count for word, count in word_freq.items() if count > 10}
    
    return len(common_words)