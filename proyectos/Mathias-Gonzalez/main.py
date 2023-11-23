import dataset_lector as lector
import rating_score
import similarity_score
import fuzzy_system
import pandas as pd
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--dataset", help="archivo del dataset con reviews de productos")
argParser.add_argument("-m", "--metadata", help="archivo del dataset con metadatos")
argParser.add_argument("-u", "--userid", help="id del usuario que solicita la recomendacion")

args = argParser.parse_args()


# dataset = lector.load('datasets/AMAZON_FASHION_5.json.gz')
# metadata = lector.load('datasets/meta_AMAZON_FASHION.json.gz')

if not args.dataset or not args.metadata:
    raise Exception("Debe proveer 2 argumentos, el dataset de reviews y el dataset de los metadatos")

dataset = lector.load_dataframe(args.dataset)
metadata = lector.load_metadata(args.metadata)

ratings = rating_score.calcular_rating_score(dataset)
similarity = similarity_score.calcular_similarity(dataset, metadata, ratings, args.userid)
fuzzysys = fuzzy_system.FuzzySystem()
recommendation_scores = fuzzysys.fuzzy_rec(ratings, similarity)
product_recomm = pd.merge(recommendation_scores.reset_index(), metadata.reset_index(), how='inner', on=['asin']).loc[:,["asin", "rec_score", "title"]].rename(columns={'title':'nombre_producto'})
print("Recomendaciones: ")
print(product_recomm)