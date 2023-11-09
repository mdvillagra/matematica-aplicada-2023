import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, HTML

"""
Similarity score. 
Implementar el cálculo de similaridad de ítems considerando el historial de búsquedas y 
compras realizadas por los usuarios: 

Consideré comparar los items basandome en los textos de reseñas y el summary de los usuarios para que por medio del calculo de 
similaridad por el coseno encuentre los items similares 

"""


def similarity_score_total(df):
    cosine_sim=calcular_similarity_score(df)
    resultados=calcular_similarity_general(df,cosine_sim)
    return resultados

def calcular_similarity_score(df): 
    #para hacer la similaridad entre [Texto de reseña] y [Resumen de la revisión}]
    df['full_text'] = df['reviewText'] + ' ' + df['summary']

    # Preprocesamiento de datos: Convertir texto de reseñas a vectores TF-IDF (Frecuencia de terminos - Frecuencia inversa de documentos )
    tfidf_vectorizer = TfidfVectorizer(stop_words='english') #Indica que se deben eliminar las palabras comunes del inglés (como "the", "and", "is", etc.) 
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['full_text'].values.astype('U')) #extrae las reseñas del dataframe y las convierte en un arreglo de texto.

    # Calcular la similitud del coseno entre los items
    """
    Método del coseno, obtiene valores numéricos entre -1 y 1, donde:
    1 indica una similitud perfecta
    0 indica ninguna similitud
    -1 indica una similitud negativa.
    """
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim # retorna la matriz de la similidad del coseno

def calcular_similarity_general(df,cosine_sim):
    resultados = [] # inicializar una lista para almacenar los resultados de la comparacion de la similitudes
    # Crear un DataFrame con la matriz cosine_sim y los valores de asin (ID del producto) como índice y columnas
    cosine_sim_df = pd.DataFrame(cosine_sim, index=df['asin'], columns=df['asin'])
    # Iterar sobre las filas de cosine_sim_df y guardar los resultados en la lista
    for idx, row in cosine_sim_df.iterrows():
        # Filtrar las similitudes que no sean 1 (similitud consigo mismo) y ordenar de mayor a menor
        similar_products = row.drop(index=idx).sort_values(ascending=False)
        # Obtener el resultado y el asin del producto actual
        result = similar_products.iloc[0]
        similar_asin = similar_products.index[0]
        # Agregar el resultado y el asin del producto actual a la lista de resultados
        resultados.append((result, similar_asin))

    # Mostrar los resultados almacenados en la lista
    i=0 # imprimir 20 como ejemplos: APARTADO PARA LIMITAR LAS IMPRESIONES 
    for resultado in resultados:
        # -------Truncar--------->
        if(i > 20): 
            print(".........................................")
            print("truncado")
            break
        i+=1
        print(f"similarity_score: {resultado[0]:.3f}, asin: {resultado[1]}")
        # ----------------------->
    print("MATRIZ :_________________________________________________________________________ ")
    # Mostrar el DataFrame con la matriz cosine_sim y los valores de asin
    display(cosine_sim_df)
    return resultados








"""
def similarity_score_un_item(df,indice_item):
    cosine_similarities=calcular_similarity_score(df)
    get_similarity_item(indice_item, cosine_similarities,df)
"""
"""
# Función para obtener recomendaciones basadas en la similitud del coseno
def get_recommendations(item_index, cosine_similarities):
    sim_scores = list(enumerate(cosine_similarities[item_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_items = sim_scores[1:11]  # Excluyendo el mismo item (similitud perfecta de 1.0)
    return [i[0] for i in top_items]


def get_similarity_item(indice_item, cosine_similarities,df):
    # Obtener recomendaciones para el primer item en el dataset
    #indice_item = 0
    coseno_sim=cosine_similarities
    recomendaciones = get_recommendations(indice_item, coseno_sim)
    subset_df = df.iloc[indice_item:indice_item+1]
    display(f"Prueba con respecto al item {indice_item}")
    display(f"Recomendacion con respecto al item {indice_item} :")
    display(subset_df)
    # Mostrar los índices de los items recomendados
    print("Recomendaciones para el item con índice", indice_item, ":")
    # Truncar 
    # pd.set_option('display.max_rows', 10)
    # Imprimir las filas de recomendaciones
    # asin -> id del producto
    # Imprimir las similaridad de un item 
    pd.set_option('display.max_colwidth', 30)
    for indice_recomendado in recomendaciones:
        fila_recomendada = df.iloc[indice_recomendado:indice_recomendado+1]
        display(fila_recomendada)
        display('')  
"""