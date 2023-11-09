


from utils import *

# Función principal
def main():
    # Define la ruta al archivo de reviews

    ## se definen las categorias : 
    
    reviews_file_path = '../Gift_Cards.json'
    
    preprocessed_reviews = load_and_preprocess_reviews(reviews_file_path) 
    ## MODULO 1 , LECTURA DE DATOS Y PREPROCESAMIENTO
    print ('Reviews dataset: MODULO 1\n\n\n')
    print (preprocessed_reviews.head())
    print('Similarity matrix: MODULO 2 \n\n\n')

    ## MODULO 3 , CALCULO DE RATING SCORE
    preprocessed_reviews = calculate_rating_score(preprocessed_reviews)
    products = preprocessed_reviews[['asin', 'rating_score']].drop_duplicates(subset=['asin'], keep='first')
    grouped_reviews = preprocessed_reviews.groupby('asin')['full_review'].apply(list)

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_vectorizer.fit(preprocessed_reviews['full_review'])

    product_similarity_matrix = calculate_similarity_between_products(grouped_reviews, tfidf_vectorizer)
    print (product_similarity_matrix)
    asin_list = grouped_reviews.index.tolist()



    # Similitud entre dos productos específicos.
    # similarity_score = product_similarity_matrix[product_idx1, product_idx2]
    # print(f"La similitud entre los productos con ASIN {asin_list[product_idx1]} y {asin_list[product_idx2]} es: {similarity_score}")

    print ('Products with rating score:')
    print(products.head())
    all_product_recommendations = get_recommendations_for_all_products(products, product_similarity_matrix)

    # Guardar la salida en un archivo JSON
    # MODULO 4 Y 5 
    for product in all_product_recommendations:
        for recommendation in product["recommendations"]:
            score = recommendation["status"]
            category, confidence = classify_recommendation(score)
            recommendation["status_def"] = f"{category} ({confidence:.2f})"


    with open('product_recommendations.json', 'w') as json_file:
        json.dump(all_product_recommendations, json_file, indent=2)

    print("Las recomendaciones de productos han sido guardadas en 'product_recommendations.json'.")

    # print(' distinc words : ', count_distinct_words(reviews_file_path))
    # print(' words more than 10 : ', words_more_than_10(reviews_file_path))

# Ejecutar el script
if __name__ == "__main__":
    main()


### build a function that read an file and return the quantity of distinct words in the file
