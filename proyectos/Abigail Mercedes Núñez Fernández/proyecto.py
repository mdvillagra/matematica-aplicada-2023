import json
import gzip
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.metrics.pairwise import cosine_similarity

'''Este código lee los datos en dataframe de pandas.'''


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


'''quitamos los duplicados de los asin en el dataframe conservando el primero y ordenando los indices 
nuevamente(utilizado en el dataframe de los productos especificamente)'''


def remove_duplicates_asin(dataframe):
    '''para evitar la advertencia del mode.chained_assignment donde nos dice que estamos modificando los datos'''
    with pd.option_context('mode.chained_assignment', None):
        dataframe.sort_values('asin', inplace=True)
        dataframe.drop_duplicates('asin', keep="first", inplace=True)
        dataframe.sort_index(inplace=True)
    return dataframe


'''modulo 1: Lector de datasets
se leen la revisión y los metadatos del producto de la categoría extrayendo de ellos 
los datos necesarios en un respectivo dataframe
'''


def modulo1(path_reviews, path_products, target_user):
    df_reviews = getDF(path_reviews)
    df_products = getDF(path_products)
    productsDataFrame = df_products[['asin', 'also_buy', 'also_view']]
    reviewsDataFrame = df_reviews[['overall', 'reviewerID', 'asin', 'reviewerName']]
    productsDataFrame = remove_duplicates_asin(productsDataFrame)

    '''Tambien extraemos el nombre del usuario objetivo para los resultados finales'''
    name_user = reviewsDataFrame[reviewsDataFrame.reviewerID == target_user].iloc[0, 3]

    return reviewsDataFrame, productsDataFrame, name_user


'''modulo 2: Rating score
calculamos el rating score (productos con el cual haya interactuado el usuario objetivo) con la
siguiente formula PR = (Nro de reviews * Sentiment Score) / (Overall * Total reviews)
'''


def modulo2(target_user, reviewsDataFrame, productsDataFrame):
    '''quitamos los duplicados de productsDataFrame'''

    productsDataFrame = remove_duplicates_asin(productsDataFrame)

    '''extraemos un dataframe con los reviews del usuario objetivo'''

    NoOfReviews = reviewsDataFrame[reviewsDataFrame['reviewerID'] == target_user]
    SentimentScore = 1

    '''creamos el dataframe que va a contener el asin del producto con su respectivo product rating'''
    productRatingDataFrame = pd.DataFrame()

    '''sacamos la cantidad de productos usando la funcion .shape[0] que especificamente devuelve el numero de 
    filas del dataframe'''

    NoOfProducts = productsDataFrame.shape[0]

    '''vemos el rating de cada producto con respecto al target_user'''
    asin_list = []
    PR_list = []

    for num in range(NoOfProducts):
        '''sacamos el asin de cada producto con la funcion .iloc[numerico,numerico] que 
        especificamente devuelve el dato contenido en dicha posicion'''
        asin_product = productsDataFrame.iloc[num, 0]

        '''usamos condiciones para seguir extrayendo los datos necesario para aplicar la formula de product rating'''
        overallDataFrame = reviewsDataFrame[
            (reviewsDataFrame['reviewerID'] == target_user) & (reviewsDataFrame['asin'] == asin_product)]
        num_overall = overallDataFrame.shape[0]

        '''si no tenemos un review del target_user no lo agregamos al dataframe ya que es cero'''

        if num_overall != 0:
            sum = 0

            '''sacamos el promedio de sus calificaciones al mismo producto'''

            for numOv in range(num_overall):
                sum = sum + overallDataFrame.iloc[numOv, 0]

            Overall = sum / num_overall
            TotalNoOfReviews = reviewsDataFrame[reviewsDataFrame['asin'] == asin_product]

            '''finalmente aplicamos la formula para el rating score'''

            PR = (NoOfReviews.shape[0] * SentimentScore) / (Overall * TotalNoOfReviews.shape[0])

            '''agregamos los resultados a sus respectivas listas para luego agregarlos al dataframe'''
            asin_list.append(asin_product)
            PR_list.append(PR)

    '''agregamos los datos al dataframe'''
    productRatingDataFrame['asin'] = asin_list
    productRatingDataFrame['product_rating'] = PR_list

    '''bandera en caso de utilizar los productos con mejor product rating como indica el articulo, utilizamos 
    aquellos que superan el promedio de los mismos para ver mas resultados y variedad lo dejamos en false'''

    flag = False
    if flag == True:
        average = productRatingDataFrame['product_rating'].mean()
        productRatingDataFrame = productRatingDataFrame.drop(
            productRatingDataFrame[productRatingDataFrame['product_rating'] < average].index)
        productRatingDataFrame = productRatingDataFrame.reset_index(drop=True)

    '''retornamos un dataframe con el asin y su respectivo product rating'''
    return productRatingDataFrame


'''Devolvemos los productos similares(los asin), es decir los la concatenacion de todos los also_buy y also_view 
de los productos con los que el usuario objetivo haya interactuado(los que tienen product rating)'''


def get_similar_products(productRatingDataFrame, productsDataFrame):
    NoOfPR = productRatingDataFrame.shape[0]
    also_buy = []
    also_view = []
    for num in range(NoOfPR):
        alsoDataFrame = productsDataFrame[(productsDataFrame['asin'] == productRatingDataFrame.iloc[num, 0])]
        listaAUX = alsoDataFrame['also_buy'].tolist()
        also_buy = listaAUX[0] + also_buy
        listaAUX = alsoDataFrame['also_view'].tolist()
        also_view = listaAUX[0] + also_view

    '''los nuevos indices para el siguiente dataframe que nos devolvera la similaridad empezando con un IDEAL
    que seria el caso mas similar/favorable con el usuario objetivo'''

    total_also = ['IDEAL'] + also_buy + also_view
    total_also = pd.unique(total_also)
    total_also = total_also.tolist()

    '''eliminamos aquellos elementos que el usuario ya compro (que serian los que tienen product rating)'''
    for num in range(NoOfPR):
        if productRatingDataFrame.iloc[num, 0] in total_also:
            total_also.remove(productRatingDataFrame.iloc[num, 0])

    '''devolvemos la lista de indice a ser utilizada para calcular la similaridad con los productos con los que ya interactuo
    el usuario objetivo'''
    return total_also


'''construimos un dataframe que nos muestra si el producto aparece o no en los distintos also_buy o also_view de cada producto
con el que interactuo(los ya comprados con product rating)'''


def get_similarity(total_also, productRatingDataFrame, productsDataFrame):
    '''utilizamos los indices calculados en get_similar_products'''
    similarityDataFrame = pd.DataFrame(index=total_also)
    NoOfPR = productRatingDataFrame.shape[0]
    NoOfTotalAlso = len(total_also)

    '''sacamos individualmente el also_buy y also_view de cada producto con product rating y vemos si nuestro indice aparecio
    o no en alguna de las listas mencionadas para luego ver la similaridad con el usuario objetivo'''

    for num in range(NoOfPR):
        listSimilarity = [1]
        alsoDataFrame = productsDataFrame[(productsDataFrame['asin'] == productRatingDataFrame.iloc[num, 0])]
        listaAUX = alsoDataFrame['also_buy'].tolist()
        also_buy = listaAUX[0]
        listaAUX = alsoDataFrame['also_view'].tolist()
        also_view = listaAUX[0]
        for numTA in range(1, NoOfTotalAlso):
            '''1 si aparece 0 si no aparece'''
            if (total_also[numTA] in also_buy) or (total_also[numTA] in also_view):
                listSimilarity.append(1)
            else:
                listSimilarity.append(0)

        similarityDataFrame[productRatingDataFrame.iloc[num, 0]] = listSimilarity

    '''devolvemos el dataframe que sera utilizado con la funcion del coseno para calcular la similaridad'''
    return similarityDataFrame


'''modulo 3: Similarity score
recolectamos los dataframes de funciones anteriores y calculamos el cosine similarity de cada uno con el caso IDEAL'''


def modulo3(productRatingDataFrame, productsDataFrame):
    total_also = get_similar_products(productRatingDataFrame, productsDataFrame)
    NoOfTotalAlso = len(total_also)
    similarityDataFrame = get_similarity(total_also, productRatingDataFrame, productsDataFrame)
    similarityScoreDataFrame = pd.DataFrame(index=total_also[1:])
    SS_list = []
    for numTA in range(1, NoOfTotalAlso):
        SS = cosine_similarity(similarityDataFrame.loc["IDEAL":"IDEAL"],
                               similarityDataFrame.loc[total_also[numTA]:total_also[numTA]])
        SS_list.append(SS[0][0])

    '''devolvemos el dataframe final para el similarity score con su respectivo asin_product'''
    similarityScoreDataFrame["similarity_score"] = SS_list

    return similarityScoreDataFrame


'''creamos el dataframe final que contiene el similarity score y el promedio de calificaciones de los mismos'''


def get_FinalDataFrame(reviewsDataFrame, similarityScoreDataFrame):
    overall_pr_list = []
    NoFinal = similarityScoreDataFrame.shape[0]

    '''sacamos el promedio de cada producto similar por su asin'''

    for num in range(NoFinal):
        overallDataFrame = reviewsDataFrame[(reviewsDataFrame['asin'] == similarityScoreDataFrame.index[num])]
        if overallDataFrame.shape[0] == 0:
            overall_pr_list.append(0)
        else:
            average = overallDataFrame['overall'].mean()
            overall_pr_list.append(average)

    FinalDataFrame = similarityScoreDataFrame
    FinalDataFrame['overall_pr'] = overall_pr_list

    return FinalDataFrame


'''funcion de membresia para un numero difuso trapezoidal'''


def trapezoid_fuzzy_number(x, a, b, c, d):
    if a == b:
        return max(min(1, (d - x) / (d - c)), 0)
    if c == d:
        return max(min((x - a) / (b - a), 1), 0)

    return max(min((x - a) / (b - a), 1, (d - x) / (d - c)), 0)


'''modulo 5: Defusificador
utilizamos defusificacion por centroide, el centro de masa se calcula por el cociente
entre (la sumatoria del producto de y por la distribucion de masa) y (la masa total)'''


def modulo5(x, y):
    numerador = 0
    denominador = 0
    for i in range(len(y)):
        numerador = numerador + (y[i] * x[i])
    for i in range(len(y)):
        denominador = denominador + y[i]

    if denominador == 0:
        return 0
    else:
        return numerador / denominador


'''sacamos el grado de activacion(firing strengths) para cada regla difusa diferenciandola
por su variable linguistica dentro del sistema difuso'''


def get_alfas(overall, similarity):
    '''overall'''
    overall_alfas = []

    '''poor'''
    overall_alfas.append(trapezoid_fuzzy_number(overall, 0, 0, 2.3, 2.5))

    '''average'''
    overall_alfas.append(trapezoid_fuzzy_number(overall, 2.3, 2.5, 2.8, 3))

    '''good'''
    overall_alfas.append(trapezoid_fuzzy_number(overall, 2.8, 3, 3.8, 4))

    '''excellent'''
    overall_alfas.append(trapezoid_fuzzy_number(overall, 3.8, 4, 5, 5))

    '''similarity'''
    similarity_alfas = []

    '''poor'''
    similarity_alfas.append(trapezoid_fuzzy_number(similarity, -1, -1, 0.1, 0.2))

    '''average'''
    similarity_alfas.append(trapezoid_fuzzy_number(similarity, 0.1, 0.2, 0.5, 0.6))

    '''good'''
    similarity_alfas.append(trapezoid_fuzzy_number(similarity, 0.5, 0.6, 0.8, 0.9))

    '''excellent'''
    similarity_alfas.append(trapezoid_fuzzy_number(similarity, 0.8, 0.9, 1, 1))

    return overall_alfas, similarity_alfas


'''modulo 4: Sistema de inferencia difuso
consideramos un sistema difuso Mamdani SISO con 16 reglas difusas'''


def modulo4(overall, similarity):
    '''calculamos los firing strengths de cada regla difusa.'''
    overall_alfas, similarity_alfas = get_alfas(overall, similarity)

    '''calculamos el conjunto difuso de salida por una base de reglas Mamdani B'(y): n V i=1  αi ∧ Bi(y)'''
    x = np.arange(0, 100.01, 0.01)
    len_x = len(x)
    values = []
    y = []
    for n in range(len_x):
        values.clear()
        '''reglas difusas:'''

        ''' (αi(firing strengths) ∧ Bi(y)(consecuente))'''

        '''R1 overall poor - similarity poor'''
        values.append(min(overall_alfas[0], similarity_alfas[0], trapezoid_fuzzy_number(x[n], 0, 5, 40, 45)))

        '''R2 overall average - similarity poor'''
        values.append(min(overall_alfas[1], similarity_alfas[0], trapezoid_fuzzy_number(x[n], 0, 5, 40, 45)))

        '''R3 overall good - similarity poor'''
        values.append(min(overall_alfas[2], similarity_alfas[0], trapezoid_fuzzy_number(x[n], 0, 5, 40, 45)))

        '''R4 overall excellent - similarity poor'''
        values.append(min(overall_alfas[3], similarity_alfas[0], trapezoid_fuzzy_number(x[n], 0, 5, 40, 45)))

        '''R5 overall poor - similarity average'''
        values.append(min(overall_alfas[0], similarity_alfas[1], trapezoid_fuzzy_number(x[n], 0, 5, 40, 45)))

        '''R6 overall poor - similarity good'''
        values.append(min(overall_alfas[0], similarity_alfas[2], trapezoid_fuzzy_number(x[n], 0, 5, 40, 45)))

        '''R7 overall poor - similarity excellent'''
        values.append(min(overall_alfas[0], similarity_alfas[0], trapezoid_fuzzy_number(x[n], 0, 5, 40, 45)))

        '''R8 overall average - similarity average'''
        values.append(min(overall_alfas[1], similarity_alfas[1], trapezoid_fuzzy_number(x[n], 40, 45, 55, 60)))

        '''R9 overall good - similarity average'''
        values.append(min(overall_alfas[2], similarity_alfas[1], trapezoid_fuzzy_number(x[n], 40, 45, 55, 60)))

        '''R10 overall average - similarity good'''
        values.append(min(overall_alfas[1], similarity_alfas[2], trapezoid_fuzzy_number(x[n], 40, 45, 55, 60)))

        '''R11 overall excellent - similarity average'''
        values.append(min(overall_alfas[3], similarity_alfas[1], trapezoid_fuzzy_number(x[n], 55, 60, 80, 85)))

        '''R12 overall good - similarity good'''
        values.append(min(overall_alfas[2], similarity_alfas[2], trapezoid_fuzzy_number(x[n], 55, 60, 80, 85)))

        '''R13 overall average - similarity excellent'''
        values.append(min(overall_alfas[1], similarity_alfas[3], trapezoid_fuzzy_number(x[n], 55, 60, 80, 85)))

        '''R14 overall excellent - similarity good'''
        values.append(min(overall_alfas[3], similarity_alfas[2], trapezoid_fuzzy_number(x[n], 80, 85, 95, 100)))

        '''R15 overall good - similarity excellent'''
        values.append(min(overall_alfas[2], similarity_alfas[3], trapezoid_fuzzy_number(x[n], 80, 85, 95, 100)))

        '''R16 overall excellent - similarity excellent'''
        values.append(min(overall_alfas[3], similarity_alfas[3], trapezoid_fuzzy_number(x[n], 80, 85, 95, 100)))

        '''finalmente n V i=1  αi ∧ Bi(y)'''
        y.append(max(values))

    '''defusificamos el resultado difuso'''

    R = modulo5(x, y)

    '''R es el resultado discreto a ser evaluado'''

    return R, y


'''se evalua el resultado de las variables luego de pasar por la inferencia difusa y se guardan los graficos de 
cada uno para ser mostrados luego'''


def modulo4and5(FinalDataFrame):
    NoFinal = FinalDataFrame.shape[0]
    graphics = pd.DataFrame(index=FinalDataFrame.index)
    result_list = []
    y_list = []
    for num in range(NoFinal):
        value, y = modulo4(FinalDataFrame.iloc[num, 1], FinalDataFrame.iloc[num, 0])
        y_list.append(y)
        '''se evalua el resultado discreto por porcentaje de 0% a 100% recomendado considerando nuestro sistema difuso'''
        if 0 <= value < 42.5:
            result_list.append('Not recommend')
        if 42.5 <= value < 57.5:
            result_list.append('Likely to recommend')
        if 57.5 <= value < 82.5:
            result_list.append('Recommend')
        if 82.5 <= value <= 100:
            result_list.append('Highly recommend')

    FinalDataFrame['result'] = result_list
    graphics['y'] = y_list
    graphics['result'] = result_list

    '''retornamos los resultados finales'''
    return FinalDataFrame, graphics


'''funcion para mostrar los graficos resultantes del sistema difuso'''


def get_graphics(graphics):
    g = graphics.shape[0]
    for num in range(g):
        x = np.arange(0, 100 + 0.01, 0.01)
        y = graphics.iloc[num, 0]
        p = pyplot.plot(x, y, linewidth=3)
        pyplot.xlim(0, 100.01)
        pyplot.ylim(0, 1.01)
        pyplot.title("asin_product: " + graphics.index[num] + "  -->  " + graphics.iloc[num, 1])
        pyplot.show()


'''funcion para mostrar grafico del antecedente de overall'''


def show_antecedent_overall():
    x = np.arange(0, 5.01, 0.01)
    pyplot.plot(x, [trapezoid_fuzzy_number(i, 0, 0, 2.3, 2.5) for i in x], linewidth=3, label="poor")
    pyplot.plot(x, [trapezoid_fuzzy_number(i, 2.3, 2.5, 2.8, 3) for i in x], linewidth=3, label="average")
    pyplot.plot(x, [trapezoid_fuzzy_number(i, 2.8, 3, 3.8, 4) for i in x], linewidth=3, label="good")
    pyplot.plot(x, [trapezoid_fuzzy_number(i, 3.8, 4, 5, 5) for i in x], linewidth=3, label="excellent")
    pyplot.xlim(0, 5.01)
    pyplot.ylim(0, 1.01)
    pyplot.legend()
    pyplot.title("Antecedent 'Overall'")
    pyplot.show()


'''funcion para mostrar grafico del antecedente de similarity'''


def show_antecedent_similarity():
    x = np.arange(-1, 1.01, 0.01)
    pyplot.plot(x, [trapezoid_fuzzy_number(i, -1, -1, 0.1, 0.2) for i in x], linewidth=3, label="poor")
    pyplot.plot(x, [trapezoid_fuzzy_number(i, 0.1, 0.2, 0.5, 0.6) for i in x], linewidth=3, label="average")
    pyplot.plot(x, [trapezoid_fuzzy_number(i, 0.5, 0.6, 0.8, 0.9) for i in x], linewidth=3, label="good")
    pyplot.plot(x, [trapezoid_fuzzy_number(i, 0.8, 0.9, 1, 1) for i in x], linewidth=3, label="excellent")
    pyplot.xlim(-1, 1.01)
    pyplot.ylim(0, 1.01)
    pyplot.legend()
    pyplot.title("Antecedent 'Similarity'")
    pyplot.show()


'''funcion para mostrar grafico del consecuente de recomendacion'''


def show_consequent_recommendation():
    x = np.arange(0, 100.01, 0.01)
    pyplot.plot(x, [trapezoid_fuzzy_number(i, 0, 5, 40, 45) for i in x], linewidth=3, label="Not recommend")
    pyplot.plot(x, [trapezoid_fuzzy_number(i, 40, 45, 55, 60) for i in x], linewidth=3, label="Likely to recommend")
    pyplot.plot(x, [trapezoid_fuzzy_number(i, 55, 60, 80, 85) for i in x], linewidth=3, label="Recommend")
    pyplot.plot(x, [trapezoid_fuzzy_number(i, 80, 85, 95, 100) for i in x], linewidth=3, label="Highly recommend")
    pyplot.xlim(0, 100.01)
    pyplot.ylim(0, 1.01)
    pyplot.legend()
    pyplot.title("Consequent 'Recommendation'")
    pyplot.show()


'''funcion que recibe: (ruta archivo reseña de productos , ruta metadatos de los producto , id usuario objetivo) 
dos rutas de archivos necesarios junto con el id del usuario objetivo para calcular la recomendacion de productos
correspondientes en base a los archivos leidos'''


def fuzzy_recommendation_system(path1, path2, target_user):
    '''modulo 1'''
    reviewsDataFrame, productsDataFrame, name_user = modulo1(path1, path2, target_user)
    print("\nUser name: ", name_user)

    '''modulo 2'''
    productRatingDataFrame = modulo2(target_user, reviewsDataFrame, productsDataFrame)

    '''modulo 3'''
    similarityScoreDataFrame = modulo3(productRatingDataFrame, productsDataFrame)

    '''obtenemos el dataframe final para trabajar con el'''
    FinalDataFrame = get_FinalDataFrame(reviewsDataFrame, similarityScoreDataFrame)

    '''modulo 4 y 5'''
    FinalDataFrame, graphics = modulo4and5(FinalDataFrame)

    '''borramos los resultados no recomendados de los graficos'''
    graphics = graphics.drop(graphics[graphics['result'] == 'Not recommend'].index)

    print("\nThe final results for (reviewerID: ", target_user, ") ", name_user, " are:")

    '''quitando los no recomendados'''
    FinalDataFrame = FinalDataFrame.drop(FinalDataFrame[FinalDataFrame['result'] == 'Not recommend'].index)
    print(FinalDataFrame)

    '''mostramos los graficos'''

    # descomentar para mostrar los graficos
    # print("\ngraphics: ")
    # graphics = graphics.drop(graphics[graphics['result'] == 'Not recommend'].index)
    # get_graphics(graphics)

    '''retornamos el dataframe final junto con sus graficos cada uno identificados por el asin'''

    return FinalDataFrame, graphics


FinalDataFrame, graphics = fuzzy_recommendation_system('Magazine_Subscriptions.json.gz',
                                                       'meta_Magazine_Subscriptions.json.gz', 'A5QQOOZJOVPSF')
'''
Algunos IDs de ususarios dentro de Magazine_Subscriptions para pruebas

ya puesto --->  A5QQOOZJOVPSF

A2GMZZ6TDYOHY7
A25MDGOMZ2GALN
A1MV5WVIZP69ZS
A2RHSC1U4FCOY9
AHD101501WCN1
'''
