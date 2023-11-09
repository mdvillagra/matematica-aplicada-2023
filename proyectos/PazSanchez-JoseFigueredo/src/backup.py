import json

# ... (tu código anterior)

def get_recommendations_for_all_products(data, similarity_matrix, top_n=10):
    product_recommendations = []
    asin_list = data['asin'].unique()
    
    for asin in asin_list:
        product_idx = data.index[data['asin'] == asin].tolist()[0]
        similarity_scores = list(enumerate(similarity_matrix[product_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        most_similar_products = similarity_scores[1:top_n+1]  # +1 porque excluimos el producto en sí
        
        recommendations = []
        for i, score in most_similar_products:
            similar_asin = data.iloc[i]['asin']
            category = determine_category(score, similarity_categories)
            recommendations.append({'product': similar_asin, 'status': category})
        
        product_recommendations.append({'product': asin, 'recommendations': recommendations})
    
    return product_recommendations

# Función principal
def main():
    # ... (tu código anterior)

    preprocessed_reviews = load_and_preprocess_reviews(reviews_file_path)
    products = preprocessed_reviews[['asin', 'rating_score']].drop_duplicates(subset=['asin'], keep='first')

    # Ajustar y transformar con TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_reviews['full_review'])

    # Calcular la matriz de similitud de productos
    product_similarity_matrix = cosine_similarity(tfidf_matrix)

    # Obtener recomendaciones para todos los productos
    all_product_recommendations = get_recommendations_for_all_products(products, product_similarity_matrix)

    # Convertir a JSON y guardar en una variable
    product_with_recommendation_json = json.dumps(all_product_recommendations, indent=2)
    print(product_with_recommendation_json)

# Ejecutar el script
if __name__ == "__main__":
    main()




# ... (tu código anterior)

def main():
    # ... (tu código anterior hasta obtener all_product_recommendations)
    
    # Obtener recomendaciones para todos los productos
    all_product_recommendations = get_recommendations_for_all_products(products, product_similarity_matrix)

    # Guardar la salida en un archivo JSON
    with open('product_recommendations.json', 'w') as json_file:
        json.dump(all_product_recommendations, json_file, indent=2)

    print("Las recomendaciones de productos han sido guardadas en 'product_recommendations.json'.")

# Ejecutar el script
if __name__ == "__main__":
    main()


### esto es para tener nomas y que no se pierda despues...