from lectura_datasets import df_reviews, df_meta
from rating_score import product_ratings
import time
from inferencia import sistema_inferencia

# print("##########################################")
# print("SIMILARITY SCORE")
# print("##########################################")


# for principal que calcula los distintos similarity scores de cada producto
# El enfoque utilizado para el calculo fue:
# 1. Se obtienen los productos en el top de los ratings calculados en "rating_score.py"
# 2. Iteramos los productos del top en 2 for anidados
# 3. En cada 'for' obtenemos los campos de "also_view", "also_buy" y "category" de cada producto
# 4. Cada campo es introducido en un conjunto (set) para eliminar duplicados
# 5. Finalmente para aplicar la similaridad de Jaccard se calcula longitud de la intersección y la unión de los conjuntos
# 6. Se calcula el similarity score usando similaridad de Jaccard
# 7. Se calcula el recommendation score

def get_recomendaciones():
    # Ordenamos los productos por rating
    product_ratings.sort_values(ascending=False, inplace=True)
    pr_dict = product_ratings.to_dict()
    # Quitamos los duplicados
    df_meta_modif = df_meta.drop_duplicates(subset=['asin'], keep='first')

    time1 = time.time()
    productos_calculados = set()
    recommendation_scores = {}
    i = 0
    for key1 in pr_dict:
        if i == 50:
            break
        i += 1
        resultado = df_meta_modif[df_meta_modif['asin'] == key1]
        if len(resultado) == 1:
            p1_view = resultado["also_view"].tolist()[0]
            p1_buy = resultado["also_buy"].tolist()[0]
            p1_cat = resultado["category"].tolist()[0]
            set_similar1 = set(p1_view)
            set_similar1.update(p1_buy)
            set_similar1.update(p1_cat)
        else:
            continue
        j = 0
        for key2 in pr_dict:
            if j == 100:
                break
            j += 1
            if key1 != key2:
                resultado2 = df_meta_modif[df_meta_modif['asin'] == key2]
                if len(resultado2) == 1:
                    p2_view = resultado2["also_view"].tolist()[0]
                    p2_buy = resultado2["also_buy"].tolist()[0]
                    p2_cat = resultado2["category"].tolist()[0]
                    set_similar2 = set(p2_view)
                    set_similar2.update(p2_buy)
                    set_similar2.update(p2_cat)
                else:
                    continue

                val_inter = len(set_similar1.intersection(set_similar2))
                val_union = len(set_similar1.union(set_similar2))

                if val_union > 0 and val_inter > 0:
                    res = val_inter/val_union
                    sistema_inferencia.input['rating'] = pr_dict[key1]
                    sistema_inferencia.input['similarity'] = res*10
                    sistema_inferencia.compute()

                    output = sistema_inferencia.output['recommendation']
                    if key2 not in productos_calculados:
                        productos_calculados.add(key2)
                        recommendation_scores[key2] = output
                    else:
                        if recommendation_scores[key2] < output:
                            recommendation_scores[key2] = output

                    # print(f"Rating: {pr_dict[key1]} Similarity: {res*10} Producto {key1}: ", sistema_inferencia.output['recommendation'])
                else:
                    sistema_inferencia.input['rating'] = pr_dict[key1]
                    sistema_inferencia.input['similarity'] = 0
                    sistema_inferencia.compute()

                    # print(f"Rating: {pr_dict[key1]} Similarity: 0 Producto {key1}: ", sistema_inferencia.output['recommendation'])

    for key in recommendation_scores:
        if recommendation_scores[key] >= 1:
            print(f"Producto {key}: ", recommendation_scores[key])

    time2 = time.time()
    print("Tiempo: ", time2 - time1)
