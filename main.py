import pandas as pd
import re
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import os
import nltk
from tabulate import tabulate
from afinn import Afinn
from tqdm import tqdm

# Inicializar tqdm para mostrar el progreso
tqdm.pandas()

# Descargar y configurar VADER lexicon
nltk.download('vader_lexicon', download_dir=os.getcwd())
nltk.data.path.append(os.getcwd())
os.system('cls||clear')

# Iniciar medición de tiempo
start_time = time.perf_counter()

# Cargar los datos
tweets = pd.read_csv('input_large.csv', nrows=1000)

# Módulo 1: Procesamiento del texto
def process_text_data(data):
    """
    Procesa los datos de texto eliminando caracteres especiales, URLs y reemplazando contracciones comunes.
    Args:
        data (pandas.DataFrame): DataFrame que contiene una columna 'sentence' con los textos a procesar.
    Returns:
        pandas.DataFrame: DataFrame con la columna 'sentence' procesada.
    """
    def clean(phrase):
        """
        Elimina caracteres especiales y URLs, y reemplaza contracciones comunes en una frase.
        Args:
            phrase (str): La frase que se va a procesar.
        Returns:
            str: La frase procesada con contracciones reemplazadas y caracteres especiales eliminados.
        """
        # Eliminar caracteres especiales y URLs
        phrase = re.sub(r"@", "", phrase)
        phrase = re.sub(r"http\S+", "", phrase)
        phrase = re.sub(r"#", "", phrase)
        
        # Reemplazar palabras con contracciones
        phrase = re.sub(r"i\'m", "i am", phrase)
        phrase = re.sub(r"i m", "i am", phrase)
        phrase = re.sub(r"you\'ll", "you will", phrase)
        phrase = re.sub(r"you ll", "you will", phrase)
        phrase = re.sub(r"i\'ve", "i have", phrase)
        phrase = re.sub(r"i ve", "i have", phrase)
        phrase = re.sub(r"i\'d", "i would", phrase)
        phrase = re.sub(r"i d", "i would", phrase)
        phrase = re.sub(r"it\'s", "it is", phrase)
        phrase = re.sub(r"it s", "it is", phrase)
        phrase = re.sub(r"he\'s", "he is", phrase)
        phrase = re.sub(r"he s", "he is", phrase)
        phrase = re.sub(r"she\'s", "she is", phrase)
        phrase = re.sub(r"she s", "she is", phrase)
        phrase = re.sub(r"that\'s", "that is", phrase)
        phrase = re.sub(r"that s", "that is", phrase)
        phrase = re.sub(r"there\'s", "there is", phrase)
        phrase = re.sub(r"there s", "there is", phrase)
        phrase = re.sub(r"what\'s", "what is", phrase)
        phrase = re.sub(r"what s", "what is", phrase)
        phrase = re.sub(r"who\'s", "who is", phrase)
        phrase = re.sub(r"who s", "who is", phrase)
        phrase = re.sub(r"where\'s", "where is", phrase)
        phrase = re.sub(r"where s", "where is", phrase)
        phrase = re.sub(r"when\'s", "when is", phrase)
        phrase = re.sub(r"when s", "when is", phrase)
        phrase = re.sub(r"why\'s", "why is", phrase)
        phrase = re.sub(r"why s", "why is", phrase)
        phrase = re.sub(r"how\'s", "how is", phrase)
        phrase = re.sub(r"how s", "how is", phrase)
        phrase = re.sub(r"let\'s", "let us", phrase)
        phrase = re.sub(r"let s", "let us", phrase)
        phrase = re.sub(r"\'em", " them", phrase)
        phrase = re.sub(r" em", " them", phrase)
        phrase = re.sub(r"\'cause", "because", phrase)
        phrase = re.sub(r" cause", "because", phrase)
        phrase = re.sub(r"\'cos", "because", phrase)
        phrase = re.sub(r" cos", "because", phrase)
        phrase = re.sub(r"\'til", "until", phrase)
        phrase = re.sub(r" til", "until", phrase)
        phrase = re.sub(r"\'bout", "about", phrase)
        phrase = re.sub(r" bout", "about", phrase)
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"won t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"can t", "can not", phrase)
        
        # Reemplazar contracciones comunes
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r" t ", " not ", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r" re ", " are ", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r" s ", " is ", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r" d ", " would ", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r" ll ", " will ", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r" ve ", " have ", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r" m ", " am ", phrase)
        phrase = re.sub(r"\'n'", " and", phrase)
        phrase = re.sub(r" n ", " and", phrase)
        return phrase
    
    # Aplicar la función clean a la columna 'sentence' para eliminar caracteres especiales, URLs y reemplazar contracciones
    tqdm.pandas(desc="Procesando texto (Limpiando caracteres y reemplazando contracciones)")
    data['sentence'] = data['sentence'].progress_apply(lambda x: clean(x.lower()))
    return data

# Módulo 2: Análisis de sentimiento usando VADER y AFINN
def analyze_sentiment(data):
    """
    Analiza el sentimiento de las oraciones en el DataFrame dado usando VADER y AFINN.
    Args:
        data (DataFrame): Un DataFrame que contiene una columna 'sentence' con las oraciones a analizar.
    Returns:
        DataFrame: El DataFrame original con tres nuevas columnas: 'pos', 'neg', 'neu' (VADER) y 
                   dos nuevas columnas 'afinn_pos' y 'afinn_neg' (AFINN).
    """
    # Inicializar el analizador de sentimientos VADER y AFINN
    sid = SentimentIntensityAnalyzer()
    afinn = Afinn()

    # Calcular los puntajes de sentimiento para cada oración
    tqdm.pandas(desc="Calculando puntajes positivos de VADER")
    data['vader_pos'] = data['sentence'].progress_apply(lambda x: sid.polarity_scores(x)['pos'])
    
    tqdm.pandas(desc="Calculando puntajes negativos de VADER")
    data['vader_neg'] = data['sentence'].progress_apply(lambda x: sid.polarity_scores(x)['neg'])
    
    tqdm.pandas(desc="Calculando puntajes neutrales de VADER")
    data['vader_neu'] = data['sentence'].progress_apply(lambda x: sid.polarity_scores(x)['neu'])
    
    # Calcular puntajes de AFINN
    tqdm.pandas(desc="Calculando puntajes de AFINN")
    data['afinn_score'] = data['sentence'].progress_apply(lambda x: afinn.score(x))
    
    # Determinar si el puntaje de AFINN es positivo o negativo
    tqdm.pandas(desc="Calculando puntajes positivos de AFINN")
    data['afinn_pos'] = data['afinn_score'].progress_apply(lambda x: x if x > 0 else 0)
    
    tqdm.pandas(desc="Calculando puntajes negativos de AFINN")
    data['afinn_neg'] = data['afinn_score'].progress_apply(lambda x: abs(x) if x < 0 else 0)
    
    return data

# Módulo 3: Aplicar las reglas fuzzy (adaptado para AFINN o VADER)
def fuzzification(pos_score, neg_score, algorithm, p_lo, p_md, p_hi, n_lo, n_md, n_hi, op_neg, op_neu, op_pos):
    """
    Aplica reglas difusas para calcular una agregación de activaciones, considerando el algoritmo especificado (VADER o AFINN).
    Args:
        pos_score (float): Puntuación positiva.
        neg_score (float): Puntuación negativa.
        algorithm (str): El algoritmo que se está utilizando, puede ser 'vader' o 'afinn'.
        Otros parámetros: Funciones de membresía fuzzy.
    Returns:
        array: Agregación de todas las activaciones.
    """
    # Establecemos el rango dependiendo del algoritmo: VADER (0 a 1), AFINN (0 a 10)
    score_range = np.arange(0, 1, 0.1) if algorithm == "vader" else np.arange(0, 10, 1)
    
    # Interpolación de la membresía para el puntaje positivo
    p_level_lo = fuzz.interp_membership(score_range, p_lo, pos_score)
    p_level_md = fuzz.interp_membership(score_range, p_md, pos_score)
    p_level_hi = fuzz.interp_membership(score_range, p_hi, pos_score)
    
    # Interpolación de la membresía para el puntaje negativo
    n_level_lo = fuzz.interp_membership(score_range, n_lo, neg_score)
    n_level_md = fuzz.interp_membership(score_range, n_md, neg_score)
    n_level_hi = fuzz.interp_membership(score_range, n_hi, neg_score)
    
    # Reglas difusas
    active_rule1 = np.fmin(p_level_lo, n_level_lo)
    active_rule2 = np.fmin(p_level_md, n_level_lo)
    active_rule3 = np.fmin(p_level_hi, n_level_lo)
    active_rule4 = np.fmin(p_level_lo, n_level_md)
    active_rule5 = np.fmin(p_level_md, n_level_md)
    active_rule6 = np.fmin(p_level_hi, n_level_md)
    active_rule7 = np.fmin(p_level_lo, n_level_hi)
    active_rule8 = np.fmin(p_level_md, n_level_hi)
    active_rule9 = np.fmin(p_level_hi, n_level_hi)
    
    # Agregación de reglas
    n2 = np.fmax(active_rule4, np.fmax(active_rule7, active_rule8))
    op_activation_lo = np.fmin(n2, op_neg)

    neu2 = np.fmax(active_rule1, np.fmax(active_rule5, active_rule9))
    op_activation_md = np.fmin(neu2, op_neu)

    p2 = np.fmax(active_rule2, np.fmax(active_rule3, active_rule6))
    op_activation_hi = np.fmin(p2, op_pos)
    
    # Agregar todas las activaciones
    aggregated = np.fmax(op_activation_lo, np.fmax(op_activation_md, op_activation_hi))
    
    return aggregated

# Módulo 4: Definir las funciones de membresía fuzzy
def define_fuzzy_membership():
    """
    Define las funciones de membresía fuzzy para varias variables.
    Args:
        None
    Returns:
        x_p (numpy.ndarray): Rango de la variable p.
        x_n (numpy.ndarray): Rango de la variable n.
        x_op (numpy.ndarray): Rango de la variable op.
        p_lo (numpy.ndarray): Función de membresía fuzzy para p baja.
        p_md (numpy.ndarray): Función de membresía fuzzy para p media.
        p_hi (numpy.ndarray): Función de membresía fuzzy para p alta.
        n_lo (numpy.ndarray): Función de membresía fuzzy para n baja.
        n_md (numpy.ndarray): Función de membresía fuzzy para n media.
        n_hi (numpy.ndarray): Función de membresía fuzzy para n alta.
        op_neg (numpy.ndarray): Función de membresía fuzzy para op negativa.
        op_neu (numpy.ndarray): Función de membresía fuzzy para op neutral.
        op_pos (numpy.ndarray): Función de membresía fuzzy para op positiva.
    """
    # Rango de las variables
    x_p = np.arange(0, 1, 0.1)
    x_n = np.arange(0, 1, 0.1)
    x_op = np.arange(0, 10, 1)

    # Funciones de membresía fuzzy
    p_lo = fuzz.trimf(x_p, [0, 0, 0.5])
    p_md = fuzz.trimf(x_p, [0, 0.5, 1])
    p_hi = fuzz.trimf(x_p, [0.5, 1, 1])
    
    n_lo = fuzz.trimf(x_n, [0, 0, 0.5])
    n_md = fuzz.trimf(x_n, [0, 0.5, 1])
    n_hi = fuzz.trimf(x_n, [0.5, 1, 1])
    
    op_neg = fuzz.trimf(x_op, [0, 0, 5])
    op_neu = fuzz.trimf(x_op, [0, 5, 10])
    op_pos = fuzz.trimf(x_op, [5, 10, 10])
    
    return x_p, x_n, x_op, p_lo, p_md, p_hi, n_lo, n_md, n_hi, op_neg, op_neu, op_pos

# Módulo 5: Defuzzificación y clasificación
def defuzzify_and_classify(aggregated, x_op):
    """
    Desfuzzifica y clasifica el resultado basado en la salida desfuzzificada.
    Si la agregación de membresía está vacía (sin activaciones), retorna una clasificación predeterminada.
    Args:
        aggregated (array-like): El conjunto de datos agregados.
        x_op (array-like): El conjunto de valores x correspondientes.
    Returns:
        tuple: Una tupla que contiene la clasificación (str) y el valor desfuzzificado (float).
    """
    # Verificar si el área bajo la función de membresía es no nula
    if np.sum(aggregated) == 0:
        # Si no hay activación, retornar una clasificación predeterminada
        return "Neutral", 5.0  # Asignamos "Neutral" y un valor desfuzzificado de 5.0 como predeterminado
    
    # Realizar la defuzzificación si hay activaciones
    defuzzified_output = fuzz.defuzz(x_op, aggregated, 'centroid')
    
    # Clasificación basada en el valor defuzzificado
    if 0 <= defuzzified_output < 3.33:
        classification = "Negative"
    elif 3.33 <= defuzzified_output <= 6.66:
        classification = "Neutral"
    else:
        classification = "Positive"
    
    return classification, defuzzified_output

# Módulo 6: Generar un resumen de los resultados
def benchmarks_and_summary(total_time, total_fuzzy_time, total_defuzz_time, total_vader_fuzzy_time, total_vader_defuzz_time, total_afinn_fuzzy_time, total_afinn_defuzz_time, total_vader_pos_count, total_vader_neg_count, total_vader_neu_count, total_afinn_pos_count, total_afinn_neg_count, total_afinn_neu_count, total_tweets):
    """
    Genera un resumen de los resultados de clasificación y tiempos de procesamiento.
    Parámetros:
    - total_time (float): Tiempo total de ejecución en segundos.
    - total_fuzzy_time (float): Tiempo total de procesamiento fuzzy en segundos.
    - total_defuzz_time (float): Tiempo total de procesamiento defuzz en segundos.
    - total_vader_fuzzy_time (float): Tiempo de procesamiento fuzzy para VADER en segundos.
    - total_vader_defuzz_time (float): Tiempo de procesamiento defuzz para VADER en segundos.
    - total_afinn_fuzzy_time (float): Tiempo de procesamiento fuzzy para AFINN en segundos.
    - total_afinn_defuzz_time (float): Tiempo de procesamiento defuzz para AFINN en segundos.
    - total_vader_pos_count (int): Cantidad de tweets positivos según VADER.
    - total_vader_neg_count (int): Cantidad de tweets negativos según VADER.
    - total_vader_neu_count (int): Cantidad de tweets neutrales según VADER.
    - total_afinn_pos_count (int): Cantidad de tweets positivos según AFINN.
    - total_afinn_neg_count (int): Cantidad de tweets negativos según AFINN.
    - total_afinn_neu_count (int): Cantidad de tweets neutrales según AFINN.
    - total_tweets (int): Cantidad total de tweets procesados.
    Retorna:
    - None: La función imprime un resumen tabulado de los resultados.
    """
    # Crear el resumen
    summary_data = {
        "Clasificación": ["Positivos VADER", "Negativos VADER", "Neutrales VADER", 
                          "Positivos AFINN", "Negativos AFINN", "Neutrales AFINN", 
                          "Total Procesado", 
                          "Tiempo Total de Ejecución (s)", "Tiempo Total Fuzzy (s)", 
                          "Tiempo Total Defuzz (s)", "Tiempo Fuzzy VADER (s)", 
                          "Tiempo Defuzz VADER (s)", "Tiempo Fuzzy AFINN (s)", 
                          "Tiempo Defuzz AFINN (s)"],
        "Cantidad": [
            f"{total_vader_pos_count} ({(total_vader_pos_count / total_tweets) * 100:.2f}%)",
            f"{total_vader_neg_count} ({(total_vader_neg_count / total_tweets) * 100:.2f}%)",
            f"{total_vader_neu_count} ({(total_vader_neu_count / total_tweets) * 100:.2f}%)",
            f"{total_afinn_pos_count} ({(total_afinn_pos_count / total_tweets) * 100:.2f}%)",
            f"{total_afinn_neg_count} ({(total_afinn_neg_count / total_tweets) * 100:.2f}%)",
            f"{total_afinn_neu_count} ({(total_afinn_neu_count / total_tweets) * 100:.2f}%)",
            total_tweets,
            f"{total_time:.10f}",
            f"{total_fuzzy_time:.10f}",
            f"{total_defuzz_time:.10f}",
            f"{total_vader_fuzzy_time:.10f}",
            f"{total_vader_defuzz_time:.10f}",
            f"{total_afinn_fuzzy_time:.10f}",
            f"{total_afinn_defuzz_time:.10f}"
        ]
    }
    
    # Crear un DataFrame para el resumen
    summary_df = pd.DataFrame(summary_data)
    
    # Imprimir el resumen usando tabulate
    print("\nResumen de los tweets:")
    print(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False, colalign=("left", "center")))

# Auxiliar: Visualización de las funciones de membresía y el resultado defuzzificado
def visualize_memberships(aggregated, x_op, defuzzified_output, op_neg, op_neu, op_pos):
    """
    Visualiza las funciones de membresía difusas y el resultado defuzzificado.

    Args:
        aggregated (array-like): Salida agregada de las funciones de membresía.
        x_op (array-like): Valores del eje x para las funciones de membresía.
        defuzzified_output (float): Valor defuzzificado.
        op_neg (array-like): Valores de membresía para la función negativa.
        op_neu (array-like): Valores de membresía para la función neutral.
        op_pos (array-like): Valores de membresía para la función positiva.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x_op, op_neg, 'b', linestyle='--', label='Negative')
    plt.plot(x_op, op_neu, 'g', linestyle='--', label='Neutral')
    plt.plot(x_op, op_pos, 'r', linestyle='--', label='Positive')
    plt.fill_between(x_op, np.zeros_like(x_op), aggregated, facecolor='orange', alpha=0.7, label='Aggregated Output')
    plt.axvline(defuzzified_output, color='red', linestyle='--', label=f'COA = {defuzzified_output:.2f}')
    plt.title('Fuzzy Membership Functions and Defuzzified Output')
    plt.xlabel('Score')
    plt.ylabel('Membership')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función principal
def main():
    """
    Función principal que realiza el análisis de sentimiento y la clasificación fuzzy de tweets.
    Modules:
     1. process_text_data: Procesa los datos de texto según las instrucciones de la Sección 3.1.
     2. analyze_sentiment: Analiza el sentimiento de las oraciones según las instrucciones de la Sección 3.2.
     3. fuzzification: Aplica reglas difusas para calcular una agregación de activaciones según las instrucciones de la Sección 3.3.1.
     4. define_fuzzy_membership: Define las funciones de membresía fuzzy según las instrucciones de la Sección 3.3.2 y 3.3.3.
     5. defuzzify_and_classify: Desfuzzifica y clasifica el resultado basado en la salida desfuzzificada según las instrucciones de la Sección 3.3.4.
     6. benchmarks_and_summary: Genera un resumen de los resultados de clasificación y tiempos de procesamiento según las instrucciones de la Sección 3.4.
    Variables:
     - processed_data: Datos procesados del texto.
     - analyzed_data: Datos analizados con los puntajes de sentimiento.
     - x_op, p_lo, p_md, p_hi, n_lo, n_md, n_hi, op_neg, op_neu, op_pos: Parámetros de membresía fuzzy.
     - data: Lista para almacenar los datos de la tabla.
     - df: DataFrame que contiene los resultados del análisis.
     - total_time: Tiempo total de ejecución.
     - total_fuzzy_time: Tiempo total de fuzzificación.
     - total_defuzz_time: Tiempo total de defuzzificación.
     - total_vader_fuzzy_time: Tiempo total de fuzzificación para VADER.
     - total_vader_defuzz_time: Tiempo total de defuzzificación para VADER.
     - total_afinn_fuzzy_time: Tiempo total de fuzzificación para AFINN.
     - total_afinn_defuzz_time: Tiempo total de defuzzificación para AFINN.
     - total_vader_pos_count: Conteo total de tweets positivos según VADER.
     - total_vader_neg_count: Conteo total de tweets negativos según VADER.
     - total_vader_neu_count: Conteo total de tweets neutrales según VADER.
     - total_afinn_pos_count: Conteo total de tweets positivos según AFINN.
     - total_afinn_neg_count: Conteo total de tweets negativos según AFINN.
     - total_afinn_neu_count: Conteo total de tweets neutrales según AFINN.
     - total_tweets: Número total de tweets analizados.
    """
    # Módulo 1: Procesamiento del texto según las instrucciones de la Sección 3.1.
    processed_data = process_text_data(tweets)
    
    # Módulo 2: Análisis de sentimiento usando VADER y AFINN según las instrucciones de la Sección 3.2.
    analyzed_data = analyze_sentiment(processed_data)
    
    # Módulo 4: Definir las funciones de membresía fuzzy según las instrucciones de la Sección 3.3.2 y 3.3.3.
    _, _, x_op, p_lo, p_md, p_hi, n_lo, n_md, n_hi, op_neg, op_neu, op_pos = define_fuzzy_membership()
    
    # Crear una lista para almacenar los datos de la tabla
    data = []
    
    # Iterar sobre los tweets para aplicar las reglas fuzzy y la defuzzificación
    with tqdm(total=analyzed_data.shape[0], desc="Procesando tweets (Fuzzificando y Defuzzificando)") as pbar:
        for _, row in analyzed_data.iterrows():
            sentence = row['sentence']
            pos_score_vader = row['vader_pos']
            neg_score_vader = row['vader_neg']
            neu_score_vader = row['vader_neu']
            pos_score_afinn = row['afinn_pos']
            neg_score_afinn = row['afinn_neg']
            
            # Módulo 3: Aplicar la fuzzificación para VADER según las instrucciones de la Sección 3.3.1.
            time_fuzzy_vader_start = time.perf_counter()
            aggregated_vader = fuzzification(pos_score_vader, neg_score_vader, 'vader', p_lo, p_md, p_hi, n_lo, n_md, n_hi, op_neg, op_neu, op_pos)
            time_fuzzy_vader_end = time.perf_counter()
            
            # Módulo 3: Aplicar la fuzzificación para AFINN según las instrucciones de la Sección 3.3.1.
            time_fuzzy_afinn_start = time.perf_counter()
            aggregated_afinn = fuzzification(pos_score_afinn, neg_score_afinn, 'afinn', p_lo, p_md, p_hi, n_lo, n_md, n_hi, op_neg, op_neu, op_pos)
            time_fuzzy_afinn_end = time.perf_counter()
            
            # Módulo 5: Defuzzificación y clasificación para VADER según las instrucciones de la Sección 3.3.4.
            time_defuzz_vader_start = time.perf_counter()
            classification_vader, defuzzified_score_vader = defuzzify_and_classify(aggregated_vader, x_op)
            time_defuzz_vader_end = time.perf_counter()
            
            # Módulo 5: Defuzzificación y clasificación para AFINN según las instrucciones de la Sección 3.3.4.
            time_defuzz_afinn_start = time.perf_counter()
            classification_afinn, defuzzified_score_afinn = defuzzify_and_classify(aggregated_afinn, x_op)
            time_defuzz_afinn_end = time.perf_counter()
            
            # Añadir los datos a la lista (incluyendo tanto los puntajes de VADER como los de AFINN)
            data.append([
                sentence,
                classification_vader,
                pos_score_vader,
                neg_score_vader,
                neu_score_vader,
                defuzzified_score_vader,
                time_fuzzy_vader_start,
                time_fuzzy_vader_end,
                round(time_fuzzy_vader_end - time_fuzzy_vader_start, 10),
                time_defuzz_vader_start,
                time_defuzz_vader_end,
                round(time_defuzz_vader_end - time_defuzz_vader_start, 10),
                classification_afinn,
                pos_score_afinn,
                neg_score_afinn,
                defuzzified_score_afinn,
                time_fuzzy_afinn_start,
                time_fuzzy_afinn_end,
                round(time_fuzzy_afinn_end - time_fuzzy_afinn_start, 10),
                time_defuzz_afinn_start,
                time_defuzz_afinn_end,
                round(time_defuzz_afinn_end - time_defuzz_afinn_start, 10)
            ])
            
            # Visualización de las membresías y resultados para VADER
            #visualize_memberships(aggregated_vader, x_op, defuzzified_score_vader, op_neg, op_neu, op_pos)
            
            # Visualización de las membresías y resultados para AFINN
            #visualize_memberships(aggregated_afinn, x_op, defuzzified_score_afinn, op_neg, op_neu, op_pos)
            
            # Actualizar el progreso
            pbar.update(1)
        
    # Crear un DataFrame con los datos
    df = pd.DataFrame(data, columns=["Tweet", "Tipo VADER", "Pos VADER", "Neg VADER", "Neu VADER", "Defuzz VADER", "Fuzzy VADER start T(s)", "Fuzzy VADER end T(s)", "Fuzzy VADER delta T(s)", "Defuzz VADER start T(s)", "Defuzz VADER end T(s)", "Defuzz VADER delta T(s)",
                                              "Tipo AFINN", "Pos AFINN", "Neg AFINN", "Defuzz AFINN", "Fuzzy AFINN start T(s)", "Fuzzy AFINN end T(s)", "Fuzzy AFINN delta T(s)", "Defuzz AFINN start T(s)", "Defuzz AFINN end T(s)", "Defuzz AFINN delta T(s)"])
    # Calcular el tiempo total de ejecución
    total_tweets = len(df)
    total_time = time.perf_counter() - start_time
    total_fuzzy_time = df["Fuzzy VADER delta T(s)"].sum() + df["Fuzzy AFINN delta T(s)"].sum()
    total_defuzz_time = df["Defuzz VADER delta T(s)"].sum() + df["Defuzz AFINN delta T(s)"].sum()
    total_vader_fuzzy_time = df["Fuzzy VADER delta T(s)"].sum()
    total_vader_defuzz_time = df["Defuzz VADER delta T(s)"].sum()
    total_afinn_fuzzy_time = df["Fuzzy AFINN delta T(s)"].sum()
    total_afinn_defuzz_time = df["Defuzz AFINN delta T(s)"].sum()
    total_vader_pos_count = (df["Tipo VADER"] == "Positive").sum()
    total_vader_neg_count = (df["Tipo VADER"] == "Negative").sum()
    total_vader_neu_count = (df["Tipo VADER"] == "Neutral").sum()
    total_afinn_pos_count = (df["Tipo AFINN"] == "Positive").sum()
    total_afinn_neg_count = (df["Tipo AFINN"] == "Negative").sum()
    total_afinn_neu_count = (df["Tipo AFINN"] == "Neutral").sum()
    
    # Guardar el DataFrame en un archivo CSV
    df.to_csv('output.csv', index=False)
    
    print("Resultados del análisis de sentimiento (head):")
    # Modificar los nombres de las columnas con saltos de línea para ajuste de texto
    df = df.drop(columns=["Fuzzy VADER start T(s)", "Fuzzy VADER end T(s)", "Fuzzy VADER delta T(s)", "Defuzz VADER start T(s)", "Defuzz VADER end T(s)", "Defuzz VADER delta T(s)", "Fuzzy AFINN start T(s)", "Fuzzy AFINN end T(s)", "Fuzzy AFINN delta T(s)", "Defuzz AFINN start T(s)", "Defuzz AFINN end T(s)", "Defuzz AFINN delta T(s)"])
    df.columns = df.columns.str.replace(' ', '\n')
    print(tabulate(df.head(10), headers="keys", tablefmt="grid", showindex=False, colalign=("left", "center", "center", "center", "center", "center", "center", "center", "center", "center"), stralign="left", maxcolwidths=[100, 10, 10, 10, 10, 10, 10, 10, 10, 10]))
    
    # Módulo 6: Generar un resumen de los resultados
    benchmarks_and_summary(total_time, total_fuzzy_time, total_defuzz_time, total_vader_fuzzy_time, total_vader_defuzz_time, total_afinn_fuzzy_time, total_afinn_defuzz_time, total_vader_pos_count, total_vader_neg_count, total_vader_neu_count, total_afinn_pos_count, total_afinn_neg_count, total_afinn_neu_count, total_tweets)

# Ejecutar la función principal
if __name__ == "__main__":
    main()