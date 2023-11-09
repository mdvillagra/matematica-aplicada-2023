# Se importan las librerías que se van a utilizar
# Se utiliza el comando: pip install pandas json gzip numpy scikit-fuzzy

import pandas as pd # Se utiliza para leer los dataframes
import json # Se utiliza para leer datos json
import gzip # Se utiliza para abrir los dataframes
import numpy as np # Se utiliza para definir los conjuntos difusos
import skfuzzy as fuzz # Se utiliza para manejar los conjuntos difusos
from skfuzzy import control as ctrl # Se utiliza para crear el sistema de inferencia


## Este código se encuentra en la página de donde se descargan los datos y se reutiliza acá para poder leer los datos
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

## Se guardan los datos de metadata y reviews que nos ayudarán a realizar los cálculos de similarity score y rating score
dfMetadata = getDF('meta_Magazine_Subscriptions.json.gz')
dfReview = getDF('Magazine_Subscriptions.json.gz')

def jaccard_similarity(set1, set2):
    """
    Esta función implementa el método del índice de Jaccardi para realizar el cálculo de similarity.
    Este método mide la similitud entre dos conjuntos dividiendo el tamaño de su intersección entre el tamaño de su unión.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # En caso de que no haya conjuntos para calcular, se retorna 0 para evitar el problema de la división con 0
    if union == 0:
        return 0.0
    return intersection / union

def calculate_similarity(df):
    """
    Esta función realiza el cálculo de similarity score de un producto con otros 1000 productos.
    Se realiza de esta manera para simplificar el código y el tiempo de ejecución para que se pueda observar con claridad sistema.
    """

    # Almacena los similarity score entre dos productos
    similarity_scores = []

    # Almacena los productos que ya han sido calculados para no repetir
    calculated_similarities = set()

    # El producto que se utilizará como prueba para comparar
    product_asin = "B00005N7NQ"

    # Contador de productos ya calculados, el máximo es 1000
    calculated_products = 0

    # Se busca el producto en el dataframe y se guarda
    producto1 = df[df["asin"] == product_asin]

    # Se realiza la unión de also_buy y also_view para tomar como conjunto de comparación
    also_buy1 = set(producto1["also_buy"].iloc[0])
    also_view1 = set(producto1["also_view"].iloc[0])
    combined1 = also_buy1.union(also_view1)

    # Se recorre el dataframe para extraer los productos
    for index, producto2 in df.iterrows():
        # Se guarda el producto a ser comparado
        asin2 = producto2["asin"]
        
        # Se valida que no sea el producto pivote
        if asin2 != product_asin:
            
            # Se valida que no haya sido ya calculado este producto
            if (product_asin, asin2) in calculated_similarities or (asin2, product_asin) in calculated_similarities:
                continue
            
            # Se realiza la union de also_buy y also_view del producto 2
            also_buy2 = set(producto2["also_buy"])
            also_view2 = set(producto2["also_view"])
            combined2 = also_buy2.union(also_view2)
            
            # Se invoca a la función de Jaccardi y el resultado obtenido se guarda en la lista
            similitud = jaccard_similarity(combined1, combined2)
            similarity_scores.append((product_asin, asin2, similitud))
            
            # Se guarda el producto en la lista de calculados
            calculated_similarities.add((product_asin, asin2))
        
            # Se incrementa el contador de productos calculados
            calculated_products += 1
        
        # Si ya se llegaron a los mil productos, se termina el ciclo
        if calculated_products >= 1000:
            break
        
    # Se retorna la lista de los score
    return similarity_scores
    
def calculate_rating(df):
    """
    Esta función realiza el cálculo de rating score de un producto.
    Se toma el promedio de los puntajes que tuvo en las reseñas el producto y se retorna una lista de los puntajes.
    """
    promedio_reviews = df.groupby('asin')['overall'].mean()
    lista_promedios = [(asin, promedio) for asin, promedio in promedio_reviews.items()]
    
    return lista_promedios

def recommendation_system(ratings, similarities):
    """
    Esta función realiza el sistema de recomendaciones.
    Se utiliza Scikit Fuzzy para realizar este sistema.
    """

    # Se crean las variables difusas en donde Similarity Score tiene un rango del 0 al 1. Rating Score del 0 al 5 y el nivel de recomendación de 0 a 1.
    similarity_score = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'similarity_score')
    product_rating = ctrl.Antecedent(np.arange(0, 5.1, 0.01), 'product_rating')
    recommendation = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'recommendation')

    # Se definen las funciones de membresía para la variable difusa de similarity score
    # Para poor el rango es (0, 0.25). Para average (0.23, 0.6). Para good (0.58, 0.8). Para excellent (0.78, 1)
    similarity_score['poor'] = fuzz.trimf(similarity_score.universe, [0, 0, 0.25])
    similarity_score['average'] = fuzz.trimf(similarity_score.universe, [0.23, 0.6, 0.6])
    similarity_score['good'] = fuzz.trimf(similarity_score.universe, [0.58, 0.8, 0.8])
    similarity_score['excellent'] = fuzz.trimf(similarity_score.universe, [0.78, 1, 1])

    # Se definen las funciones de membresía para la variable difusa de rating score
    # Para poor el rango es (0, 2). Para average (1.8, 3). Para good (2.8, 4). Para excellent (3.8, 5)
    product_rating['poor'] = fuzz.trimf(product_rating.universe, [0, 0, 2])
    product_rating['average'] = fuzz.trimf(product_rating.universe, [1.8, 3, 3])
    product_rating['good'] = fuzz.trimf(product_rating.universe, [2.8, 4, 4])
    product_rating['excellent'] = fuzz.trimf(product_rating.universe, [3.8, 5, 5])

    # Se definen las funciones de membresía para la variables difusa de recommendation
    # Para poor el rango es (0, 0.25). Para average (0.23, 0.5). Para good (0.48, 0.8). Para excellent (0.78, 1)
    recommendation['not_recommended'] = fuzz.trimf(recommendation.universe, [0, 0, 0.25])
    recommendation['likely_to_recommended'] = fuzz.trimf(recommendation.universe, [0.23, 0.5, 0.5])
    recommendation['recommended'] = fuzz.trimf(recommendation.universe, [0.48, 0.8, 0.8])
    recommendation['highly_recommended'] = fuzz.trimf(recommendation.universe, [0.78, 1, 1])

    # Se definen las reglas difusas, como hay 4 funciones de membresía para rating y 4 para similarity. Se definen 16 reglas difusas.
    rule1 = ctrl.Rule(product_rating['excellent'] & similarity_score['excellent'], recommendation['highly_recommended'])
    rule2 = ctrl.Rule(product_rating['excellent'] & similarity_score['good'], recommendation['highly_recommended'])
    rule3 = ctrl.Rule(product_rating['excellent'] & similarity_score['average'], recommendation['recommended'])
    rule4 = ctrl.Rule(product_rating['excellent'] & similarity_score['poor'], recommendation['likely_to_recommended'])
    rule5 = ctrl.Rule(product_rating['good'] & similarity_score['excellent'], recommendation['highly_recommended'])
    rule6 = ctrl.Rule(product_rating['good'] & similarity_score['good'], recommendation['highly_recommended'])
    rule7 = ctrl.Rule(product_rating['good'] & similarity_score['average'], recommendation['recommended'])
    rule8 = ctrl.Rule(product_rating['good'] & similarity_score['poor'], recommendation['not_recommended'])
    rule9 = ctrl.Rule(product_rating['average'] & similarity_score['excellent'], recommendation['recommended'])
    rule10 = ctrl.Rule(product_rating['average'] & similarity_score['good'], recommendation['recommended'])
    rule11 = ctrl.Rule(product_rating['average'] & similarity_score['average'], recommendation['likely_to_recommended'])
    rule12 = ctrl.Rule(product_rating['average'] & similarity_score['poor'], recommendation['not_recommended'])
    rule13 = ctrl.Rule(product_rating['poor'] & similarity_score['excellent'], recommendation['likely_to_recommended'])
    rule14 = ctrl.Rule(product_rating['poor'] & similarity_score['good'], recommendation['not_recommended'])
    rule15 = ctrl.Rule(product_rating['poor'] & similarity_score['average'], recommendation['not_recommended'])
    rule16 = ctrl.Rule(product_rating['poor'] & similarity_score['poor'], recommendation['not_recommended'])

    # Almacena los resultados del sistema
    recommendation_results = []

    # Recorre la cantidad de productos del similarity ya que este tiene un límite que en nuestro caso es de 1000 productos
    for i in range(len(similarities)):
        asin1 = ratings[i][0]   # ASIN del producto en `ratings`
        rating = ratings[i][1]  # Valor de rating en `ratings`

        asin2 = similarities[i][1]  # ASIN del producto en `similarities`
        similarity_score = similarities[i][2]  # Valor de similitud en `similarities`

        # Se valida que sea el mismo producto
        if asin1 != asin2:
            continue

        # Se utiliza ControlSystem de Scikit Fuzz para crear el sistema de inferencia y se definen las reglas que creamos
        recommendation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16])

        # Se inicializa el simulador
        recommendation_system = ctrl.ControlSystemSimulation(recommendation_ctrl)

        # Se definen nuestros valores de entrada que en nuestro caso son el Similarity Score y Rating Score
        recommendation_system.input['similarity_score'] = similarity_score
        recommendation_system.input['product_rating'] = rating

        # Se llama el método compute donde se hacen los cálculos de las reglas, se utiliza el Mamdani Inference y se realiza la defuzzificación
        recommendation_system.compute()

        # Se obtiene el resultado defuzzificado del sistema
        recommendation_level = recommendation_system.output['recommendation']

        # Se crea un diccionario en donde se va a guardar el asin del productom, el mensaje de recomendación y la salida del defuzzificador
        result = {
            'product': asin2,
            'recommendation': '',
            'score': recommendation_level
        }

        # Se analiza en que rango se encuentra la salida del defuzzificador para poder definir el mensaje de recomendación
        if recommendation_level <= 0.25:
            result['recommendation'] = 'Not Recommended'
        elif 0.25 < recommendation_level <= 0.5:
            result['recommendation'] = 'Likely to Recommended'
        elif 0.5 < recommendation_level <= 0.8:
            result['recommendation'] = 'Recommended'
        elif recommendation_level > 0.8:
            result['recommendation'] = 'Highly Recommended'

        # Se guarda el diccionario en la lista de resultados
        recommendation_results.append(result)

    return(recommendation_results)

# Se realizan las llamadas a las funciones para realizar el sistema de inferencia
ratings = calculate_rating(dfReview)
similarities = calculate_similarity(dfMetadata)
results = recommendation_system(ratings, similarities)

# Ordenar la lista por 'score' de mayor a menor
sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

# Se muestran en pantalla los 10 productos más recomendados
for result in sorted_results[:10]:
    print(f"Product: {result['product']}, Recommendation: {result['recommendation']}, Score: {result['score']}")

