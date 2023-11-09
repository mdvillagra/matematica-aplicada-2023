import pandas as pd
import gzip
import json
from sklearn.metrics.pairwise import cosine_similarity
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import argparse
from prettytable import PrettyTable


# Cargamos cada linea del json
def parse(path):
    try:
        g = gzip.open(path, "rb")
        for l in g:
            yield json.loads(l)
    except Exception as e:
        print(e)
        return None


# Cargamos el json en un dataframe de pandas
def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def get_average_ratings(reviews):
    average_ratings = {}

    """
    Agrupamos por 'asin' que se encuentra en el json
    El asin es la id unica de cada producto
    """
    grouped_reviews = reviews.groupby("asin")
    for asin, group in grouped_reviews:
        average_rating = group[
            "overall"
        ].mean()  # Calcula el promedio por datos agrupados
        average_ratings[asin] = average_rating
    return average_ratings


def get_user_profiles(reviews):
    """
    Obtenemos los perfiles de los usuarios
    """
    user_profiles = []
    for user_id, user_data in reviews.groupby("reviewerID"):
        user_profile = pd.Series(0, index=reviews["asin"].unique())
        for index, row in user_data.iterrows():
            user_profile[row["asin"]] = 1
        user_profiles.append(user_profile)
    return user_profiles


def get_item_similarity(user_profiles):
    """
    Se retorna la similaridad entre los items
    """
    user_profiles_matrix = np.array(user_profiles)
    return cosine_similarity(user_profiles_matrix)


def index_to_asin(reviews):
    return {index: asin for index, asin in enumerate(reviews["asin"].unique())}


def asin_to_index(reviews):
    return {asin: index for index, asin in enumerate(reviews["asin"].unique())}


def get_items_reviewed_by_user(user_index, user_profiles):
    """
    Esta funcion devuelve una lista de items revisados por el usuario
    """
    if user_index >= len(user_profiles):
        raise Exception("The user does not exist")
    return [i for i, value in enumerate(user_profiles[user_index]) if value == 1]


def get_by_items_related(
    metadata,
    item_index_to_asin,
    item_asin_to_index,
    item_index,
    possible_recomendations,
    items_reviewed_by_user,
    item_similarity,
    key,
):
    """
    Este código obtiene los elementos relacionados con el elemento que se ingresó.
    Itera a través de los elementos relacionados del elemento y los agrega a los posibles
    lista de recomendaciones si aún no están en la lista y si
    el artículo aún no ha sido revisado por el usuario.
    """
    related = metadata[metadata["asin"] == item_index_to_asin[item_index]]
    for similar_id in related.iloc[0][key]:
        if similar_id in item_asin_to_index:
            if (similar_id not in possible_recomendations) and (
                similar_id not in items_reviewed_by_user
            ):
                similar_index = item_asin_to_index[similar_id]
                similar_info = metadata[metadata["asin"] == similar_id]
                similar_title = similar_info.iloc[0]["title"]
                similar_score = item_similarity[item_index][similar_index]
                similar_rating = similar_info.iloc[0]["average_rating"]

                obj = {
                    "product_id": similar_id,
                    "title": similar_title,
                    "similarity": similar_score,
                    "rating": similar_rating,
                }

                possible_recomendations.append(obj)


def get_by_items_similar(
    metadata,
    items_reviewed_by_user,
    item_similarity,
    item_index_to_asin,
    possible_recomendations,
    item_index,
):
    """
    Recorremos elementos similares a uno dado, verifica criterios y
    agrega recomendaciones potenciales (no revisadas por el usuario)
    a una lista, incluyendo información como
    identificador único, título, similitud y calificación promedio.
    """
    for similar_index in range(len(item_similarity[item_index])):
        if (
            item_similarity[item_index][similar_index] > 0
            and item_index != similar_index
        ):
            # Se revisa que el item al que se quiere acceder esté en la lista de items
            if similar_index in item_index_to_asin:
                similar_id = item_index_to_asin[similar_index]
                # Se agrega el item si no está en la lista de posibles recomendaciones
                # y si no está en la lista de items con review del usuario
                if (similar_id not in possible_recomendations) and (
                    similar_id not in items_reviewed_by_user
                ):
                    similar_info = metadata[metadata["asin"] == similar_id]
                    similar_title = similar_info.iloc[0]["title"]
                    similar_score = item_similarity[item_index][similar_index]
                    similar_rating = similar_info.iloc[0]["average_rating"]

                    obj = {
                        "product_id": similar_id,
                        "title": similar_title,
                        "similarity": similar_score,
                        "rating": similar_rating,
                    }

                    possible_recomendations.append(obj)


def get_possible_recomendations(
    items_reviewed_by_user, item_similarity, reviews, metadata
):
    item_index_to_asin = index_to_asin(reviews)
    item_asin_to_index = asin_to_index(reviews)

    # Se inicializa la lista de posibles recomendaciones
    possible_recomendations = []

    # Se buscan los items similares
    for item_index in items_reviewed_by_user:
        # Se itera por la lista de similaridades del item
        get_by_items_similar(
            metadata,
            items_reviewed_by_user,
            item_similarity,
            item_index_to_asin,
            possible_recomendations,
            item_index,
        )

        get_by_items_related(
            metadata,
            item_index_to_asin,
            item_asin_to_index,
            item_index,
            possible_recomendations,
            items_reviewed_by_user,
            item_similarity,
            "also_view",
        )

        get_by_items_related(
            metadata,
            item_index_to_asin,
            item_asin_to_index,
            item_index,
            possible_recomendations,
            items_reviewed_by_user,
            item_similarity,
            "also_buy",
        )

    return possible_recomendations


def get_recomendator():
    """
    Creamos las variables difusas,
    definimos funciones de membresia
    y definimos las reglas difusas
    """
    rating = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "rating")
    similarity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "similarity")
    recomendation = ctrl.Consequent(np.arange(0, 1.1, 0.1), "recomendation")

    rating["low"] = fuzz.trimf(rating.universe, [0, 0, 0.5])
    rating["mid"] = fuzz.trimf(rating.universe, [0, 0.5, 1])
    rating["high"] = fuzz.trimf(rating.universe, [0.5, 1, 1])

    similarity["low"] = fuzz.trimf(similarity.universe, [0, 0, 0.5])
    similarity["mid"] = fuzz.trimf(similarity.universe, [0, 0.5, 1])
    similarity["high"] = fuzz.trimf(similarity.universe, [0.5, 1, 1])

    recomendation["low"] = fuzz.trimf(recomendation.universe, [0, 0, 0.5])
    recomendation["mid"] = fuzz.trimf(recomendation.universe, [0, 0.5, 1])
    recomendation["high"] = fuzz.trimf(recomendation.universe, [0.5, 1, 1])

    rule1 = ctrl.Rule(rating["low"] & similarity["low"], recomendation["low"])
    rule2 = ctrl.Rule(rating["low"] & similarity["mid"], recomendation["low"])
    rule3 = ctrl.Rule(rating["low"] & similarity["high"], recomendation["mid"])
    rule4 = ctrl.Rule(rating["mid"] & similarity["low"], recomendation["low"])
    rule5 = ctrl.Rule(rating["mid"] & similarity["mid"], recomendation["mid"])
    rule6 = ctrl.Rule(rating["mid"] & similarity["high"], recomendation["high"])
    rule7 = ctrl.Rule(rating["high"] & similarity["low"], recomendation["mid"])
    rule8 = ctrl.Rule(rating["high"] & similarity["mid"], recomendation["high"])
    rule9 = ctrl.Rule(rating["high"] & similarity["high"], recomendation["high"])

    recomendation_sys = ctrl.ControlSystem(
        [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]
    )
    return ctrl.ControlSystemSimulation(recomendation_sys)


def main(id):
    user_index = id

    """
    Cambiar los valores por los paths 
    donde se encuentran los archivos
    """
    reviews = getDF("/home/frejala/Desktop/tp-mate-aplicada/data/Software_5.json.gz")
    metadata = getDF(
        "/home/frejala/Desktop/tp-mate-aplicada/data/meta_Software.json.gz"
    )
    average_ratings = get_average_ratings(reviews)
    metadata["average_rating"] = metadata["asin"].map(average_ratings)
    user_profiles = get_user_profiles(reviews)
    item_similarity = get_item_similarity(user_profiles)

    try:
        items_reviewed_by_user = get_items_reviewed_by_user(user_index, user_profiles)

        possible_recomendations = get_possible_recomendations(
            items_reviewed_by_user, item_similarity, reviews, metadata
        )

        recomendator = get_recomendator()
        decision_threshold = 0.8

        # Esto es solo para imprimir mas lindo
        table = PrettyTable()
        table.field_names = [
            "Title",
            "Average Rating",
            "Similarity",
            "Valor de recomendación",
        ]

        for recomendation in possible_recomendations:
            recomendator.input["rating"] = recomendation["rating"]
            recomendator.input["similarity"] = recomendation["similarity"]
            recomendator.compute()

            if recomendator.output["recomendation"] >= decision_threshold:
                table.add_row(
                    [
                        recomendation["title"],
                        "{:.2f}".format(recomendation["rating"]),
                        "{:.2f}".format(recomendation["similarity"]),
                        "{:.2f}".format(recomendator.output["recomendation"]),
                    ]
                )

        print(table)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("id", type=int, default=0, help="Indice del usuario")

    args = parser.parse_args()

    main(args.id)
