import pandas as pd
import skfuzzy as fuzz
import numpy as np
"""
El método del centroide ponderado: es un enfoque utilizado en la defusificación en lógica difusa. La defusificación es el proceso de 
convertir una salida difusa (valores difusos) en un valor concreto o nítido. En este caso, combinamos el similarity score y el 
rating score para tomar una decisión final sobre qué producto recomendar. Al calcular el centroide ponderado, estamos teniendo un 
valor concreto que representa la recomendación final, considerando tanto la similitud de contenido como las calificaciones de los usuarios.
                                n                                      n
Centroide Ponderado =           ∑ (Valor_i)x(Grado_de_Pertenencia_i) / ∑ (Grado_de_Pertenencia_i)
                                i=1                                   i=1

"""

def calcular_grados_pertenencia(similarity_scores, rating_scores):
    # Definir universos para similarity score y rating score
    similarity_universe = np.arange(0, 1.1, 0.1)  # Universo para similarity score de 0 a 1 con paso de 0.1
    rating_universe = np.arange(1, 6, 1)  # Universo para rating score de 1 a 5

    # Definir funciones de membresía para similarity score y rating score
    similarity_score_mf = fuzz.trimf(similarity_universe, [0, 0.5, 1])  # Función de membresía triangular para similarity score
    rating_score_mf = fuzz.trimf(rating_universe, [1, 3, 5])  # Función de membresía triangular para rating score

    # Calcular grados de pertenencia para similarity scores y rating scores
    similarity_memberships = [fuzz.interp_membership(similarity_universe, similarity_score_mf, score) for score in similarity_scores]
    rating_memberships = [fuzz.interp_membership(rating_universe, rating_score_mf, score) for score in rating_scores]

    return similarity_memberships, rating_memberships

def defusicador(resultados, rating_score):
    similarity_scores = [similarity_score for similarity_score, _ in resultados]
    rating_scores = [rating_score[rating_score['asin'] == asin]['overall'].values[0] for _, asin in resultados]

    # Calcular grados de pertenencia usando las funciones de membresía
    similarity_memberships, rating_memberships = calcular_grados_pertenencia(similarity_scores, rating_scores)

    # Calcular el centroide ponderado manualmente
    suma_similaridad_ponderada = np.sum(np.array(similarity_scores) * np.array(similarity_memberships))
    suma_rating_ponderada = np.sum(np.array(rating_scores) * np.array(rating_memberships))
    total_similarity_membership = np.sum(np.array(similarity_memberships))
    total_rating_membership = np.sum(np.array(rating_memberships))

    centroid_similarity_score = suma_similaridad_ponderada / total_similarity_membership 
    centroid_rating_score = suma_rating_ponderada / total_rating_membership 

    print("(Raiting Score), (Similarity Score)")
    print(f"Centroide Ponderado: ({centroid_rating_score:.4f}, {centroid_similarity_score:.4f})")
