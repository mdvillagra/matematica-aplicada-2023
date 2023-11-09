import pandas as pd


def sistema_inferencia_difusa(resultados,rating_score):
    i=0
    # Iterar sobre los resultados y aplicar las reglas
    for resultado in resultados:
        similarity_score = resultado[0]
        similar_asin = resultado[1]
        
        #----------------------------------------------------
        # Truncar N items:
        
        i+=1
        if(i > 50):
            print(".........................................")
            print("truncado")
            break
        
        #----------------------------------------------------
        """
        # Product rating score:
        Excellent   3.8 <= r <= 5
        Good        2.8 <= r <= 4
        Average     2.3 <= r <= 3
        Poor        0   <= r <= 2.5
        
        # Similarity Score
        Excellent   0.70 <= s <= 1
        Good        0.5  <= s <= 0.8
        Average     0.25 <= s <= 0.55
        Poor        0    <= s <= 0.30

        """
        # Obtener el rating score del producto actual (si existe en el DataFrame de rating_score)
        rating_score_product = rating_score[rating_score['asin'] == similar_asin]['overall'].values
        
        # Verificar las condiciones
        ###################### REGLA 1 ##########################
        # REGLA 1: IF Overall_Product_Rating= Excellent AND Past_Purchase_Similarity= Excellent
        
        if (rating_score_product[0] <= 5 and rating_score_product[0] >= 3.8)  and len(rating_score_product) > 0 and (similarity_score >= 0.70 and similarity_score <= 1):
            print(f"Recomendation_Level= Highly Recommended  similarity_score: {similarity_score:.3f}, rating_score: {rating_score_product[0]}, asin: {similar_asin}")
        
        ###################### REGLA 2 ##########################
        # REGLA 2: IF Overall_Product_Rating= Excellent AND Past_Purchase_Similarity= Good 
        elif (rating_score_product[0] <= 5 and rating_score_product[0] >= 3.8) > 0 and (similarity_score >= 0.50 and similarity_score <= 0.8):
            print(f"Recomendation_Level= Highly Recommended  similarity_score: {similarity_score:.3f}, rating_score: {rating_score_product[0]}, asin: {similar_asin}")
        
        ###################### REGLA 3 ##########################
        # REGLA 3: IF Overall_Product_Rating= Good AND Past_Purchase_Similarity= Good     
        elif (rating_score_product[0] <= 4 and rating_score_product[0] >= 2.8) and len(rating_score_product) > 0 and (similarity_score >= 0.50 and similarity_score <= 0.8):
            print(f"Recomendation_Level= Highly Recommended  similarity_score: {similarity_score:.3f}, rating_score: {rating_score_product[0]}, asin: {similar_asin}")

        ###################### REGLA 4 ##########################     
        # REGLA 4: IF Overall_Product_Rating= Average AND Past_Purchase_Similarity= Excellent    
        elif (rating_score_product[0] <= 3 and rating_score_product[0] >= 2.3) and len(rating_score_product) > 0 and (similarity_score >= 0.70 and similarity_score <= 1):
            print(f"Recomendation_Level= Highly Recommended  similarity_score: {similarity_score:.3f}, rating_score: {rating_score_product[0]}, asin: {similar_asin}")

        ###################### REGLA 5 ##########################  
        # REGLA 5: IF Overall_Product_Rating= Average AND Past_Purchase_Similarity= Good    
        elif (rating_score_product[0] <= 3 and rating_score_product[0] >= 2.3) and len(rating_score_product) > 0 and (similarity_score >= 0.50 and similarity_score <= 0.8):
            print(f"Recomendation_Level= Recommended  similarity_score: {similarity_score:.3f}, rating_score: {rating_score_product[0]}, asin: {similar_asin}")

        ###################### REGLA 6 ##########################    Average     0.25 <= s <= 0.55
        # REGLA 6: IF Overall_Product_Rating= Average AND Past_Purchase_Similarity= Average    
        elif (rating_score_product[0] <= 3 and rating_score_product[0] >= 2.3) and len(rating_score_product) > 0 and (similarity_score >= 0.25 and similarity_score <= 0.55):
            print(f"Recomendation_Level= Recommended  similarity_score: {similarity_score:.3f}, rating_score: {rating_score_product[0]}, asin: {similar_asin}")

        ###################### REGLA 7 ##########################   
        # REGLA 7: IF Overall_Product_Rating= Excellent AND Past_Purchase_Similarity= Good    
        elif (rating_score_product[0] <= 5 and rating_score_product[0] >= 3.8)  and len(rating_score_product) > 0 and (similarity_score >= 0.50 and similarity_score <= 0.8):
            print(f"Recomendation_Level= Recommended  similarity_score: {similarity_score:.3f}, rating_score: {rating_score_product[0]}, asin: {similar_asin}")

        ###################### REGLA 8 ##########################
        # REGLA 8: IF Overall_Product_Rating= Average AND Past_Purchase_Similarity= Average   
        elif (rating_score_product[0] <= 3 and rating_score_product[0] >= 2.3) and len(rating_score_product) > 0 and (similarity_score >= 0.25 and similarity_score <= 0.55):
            print(f"Recomendation_Level= Likely to recommend  similarity_score: {similarity_score:.3f}, rating_score: {rating_score_product[0]}, asin: {similar_asin}")
       
        ###################### REGLA 9 ##########################
        # REGLA 9: IF Overall_Product_Rating= Average AND Past_Purchase_Similarity= Average   
        elif (rating_score_product[0] <= 3 and rating_score_product[0] >= 2.3) and len(rating_score_product) > 0 and (similarity_score >= 0.25 and similarity_score <= 0.55):
            print(f"Recomendation_Level= Recommend  similarity_score: {similarity_score:.3f}, rating_score: {rating_score_product[0]}, asin: {similar_asin}")
   
        ###################### REGLA 10 ##########################
        # REGLA 10: IF Overall_Product_Rating= Average AND Past_Purchase_Similarity= Average   
        elif (rating_score_product[0] <= 3 and rating_score_product[0] >= 2.3) and len(rating_score_product) > 0 and (similarity_score >= 0.25 and similarity_score <= 0.55):
            print(f"Recomendation_Level= Not recommend  similarity_score: {similarity_score:.3f}, rating_score: {rating_score_product[0]}, asin: {similar_asin}")
    
        ###################### REGLA 11 ##########################
        # REGLA 11: IF Overall_Product_Rating= Poor AND Past_Purchase_Similarity= Average   
        elif (rating_score_product[0] <= 2.5 and rating_score_product[0] >= 0) and len(rating_score_product) > 0 and (similarity_score >= 0.25 and similarity_score <= 0.55):
            print(f"Recomendation_Level= Not recommend  similarity_score: {similarity_score:.3f}, rating_score: {rating_score_product[0]}, asin: {similar_asin}")

        ###################### REGLA 12 ##########################
        # REGLA 12: IF Overall_Product_Rating= Excellent AND Past_Purchase_Similarity= Good   
        elif (rating_score_product[0] <= 5 and rating_score_product[0] >= 3.8)  and len(rating_score_product) > 0 and (similarity_score >= 0.50 and similarity_score <= 0.8):
            print(f"Recomendation_Level= Not recommend  similarity_score: {similarity_score:.3f}, rating_score: {rating_score_product[0]}, asin: {similar_asin}")


