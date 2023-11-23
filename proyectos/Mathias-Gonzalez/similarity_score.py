import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def obtener_Y(ratings, n):
    # Obtener lista de recomendaciones
    y = ratings.sort_values('product_rating',ascending = False).head(n)
    return pd.Series(y.index, name="asin")

def obtener_X(dataset, metadata):
    # Combinar productos del historial con con productos relacionados
    # df_join = pd.merge(dataset.reset_index(), metadata.reset_index(), how='inner', on=['asin']).loc[:,["reviewerID", "asin", "also_buy", "also_view"]]
    df_join = pd.merge(dataset.reset_index(), metadata.reset_index(), how='inner', on=['asin']).loc[:,["asin", "also_buy", "also_view"]]
    # Remover productos relacionados vacios
    df_join = df_join[~df_join["also_buy"].isna()]
    df_join = df_join[~df_join["also_view"].isna()]
    # Combina los productos que el usuario conoce en una lista
    df_join["x"] = df_join.apply(lambda x: (x["also_buy"] or []) + (x["also_view"] or []) + [x['asin']], axis=1)
    # Distribuye la lista en columnas para mejor manipulacion y elimina duplicados
    df_join = df_join.explode("x").reset_index(drop=True)["x"].drop_duplicates(keep="first").rename("asin")
    # Filtra los productos de los que no se tiene informacion
    df_join = df_join[df_join.isin(metadata["asin"])]

    return df_join

# Calcular el cosine similarity de los productos
def calcular_similarity(dataset, metadata, ratings, userID = None):

    if userID is None:
        # Si no se especifica el ID del usuario, se selecciona un usuario aleatorio para testear
        userID = dataset.sample()["reviewerID"].iloc[0]
        print(f'Usuario seleccionado al azar: {userID}')

    nro_recomendaciones = 50 # Maximo numero de productos a recomendar
    userReviews = dataset[dataset["reviewerID"] == userID]
    history = userReviews["asin"]

    print("Productos comprados: ")
    print(pd.merge(history.reset_index(), metadata.reset_index(), how='inner', on=['asin']).loc[:,["asin", "title"]])
    
    if(len(userReviews.index) == 0):
        print("No se tiene informacion del usuario con el id especificado")
        exit()
    
    x = obtener_X(userReviews, metadata) # Productos en historial y relacionados
    y = obtener_Y(ratings, nro_recomendaciones) # Productos para recomendar

    # Filtra los productos que el usuario ya compro
    y = y[~y.isin(history)]
    y = y.drop_duplicates(keep='first')
    # Quita los productos de los que no se tengan metadatos
    y = y[y.isin(metadata["asin"])]

    if (len(x.index) == 0 or len(y.index)==0):
        print("No existen metadatos para recomendar los productos")
        exit()
    
    # Seleccionar datos para calcular similaridad    
    columnas = ['title', 'description', 'feature', 'brand']
    X = metadata[metadata["asin"].isin(x)][['asin'] + columnas].fillna('')
    Y = metadata[metadata["asin"].isin(y)][['asin'] + columnas].fillna('')

    # Formatear datos como cadenas
    X["description"] = X["description"].astype(str)
    X["feature"] = X["feature"].astype(str)
    X["brand"] = X["brand"].astype(str)
    Y["description"] = Y["description"].astype(str)
    Y["feature"] = Y["feature"].astype(str)
    Y["brand"] = Y["brand"].astype(str)

    # Guardar datos formateados
    X["doc"] = X[columnas].agg(' '.join, axis = 1) 
    Y["doc"] = Y[columnas].agg(' '.join, axis = 1) 

    docX = X["doc"]
    docY = Y["doc"]

    # Create a Vectorizer Object
    vectorizer = CountVectorizer(stop_words="english", min_df=1)
    
    vectorizer.fit(pd.concat([docX, docY]))

    # Encode the Document
    vectorX = pd.DataFrame(vectorizer.transform(docX).todense(), columns=vectorizer.get_feature_names_out(), index=X["asin"])
    vectorY = pd.DataFrame(vectorizer.transform(docY).todense(), columns=vectorizer.get_feature_names_out(), index=Y["asin"])
    
    # Mapear productos con recomendaciones
    M = pd.merge(X['asin'], Y['asin'], how="cross")

    # Calcular y guardar similarity score para 
    M["sim"] = M.apply(lambda x: cosine_similarity(vectorX.loc[[x.iloc[0]]], vectorY.loc[[x.iloc[1]]])[0][0], axis=1)
    
    M = M.rename(columns={"asin_y": "asin"})
    return M.groupby("asin")[["sim"]].max() # Solo considerar el mayor similarity score de cada producto a recomendar


# def custom_sim(row, dX, dY):
#     idx = row.iloc[0]
#     idy = row.iloc[1]

#     vx = dX.loc[[idx]].to_numpy()
#     vy = dY.loc[[idy]].to_numpy()

#     dot = np.dot(vx, vy.T)
#     norm = np.linalg.norm(vx) * np.linalg.norm(vy)

#     return dot / norm

