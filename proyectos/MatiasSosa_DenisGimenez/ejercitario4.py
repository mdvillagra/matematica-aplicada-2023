import numpy as np

def range_values(triangular_number_list):
    '''
    Calcula el rango de valores no nulos 

    Args:
        triangular_number_list (list): Lista de numeros triangulares.
        
    Returns:
        list: rango de valores con grado de pertenencia no nulos

    Example:
        >>> range_values([[2, 4, 6], [6, 4, 8]])
        array([2.        , 2.00600601, 2.01201201, 2.01801802, 2.02402402,
        ...
        7.97597598, 7.98198198, 7.98798799, 7.99399399, 8.        ])
        Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
    '''
    extremo_izq = min(triangular_number[0] for triangular_number in triangular_number_list)
    extremo_der = max(triangular_number[2] for triangular_number in triangular_number_list)

    return np.linspace(extremo_izq, extremo_der, 10)


def triangular_membership_degree(x, triangular_number):
    '''
    Calcula el grado de pertenencia de un valor x en el numero triangular(conjunto difuso).

    Args:
        x (float): Número cuyo grado de pertenencia se va a evaluar.
        triangular number (list): Numero triangular.
        
    Returns:
        float: Grado de pertenencia de x

    Example:
        >>> triangular_membership_degree(2.25, [2.0, 3.0, 4.0])
        0.250000000000000
    '''
    a, b, c = triangular_number

    return max(min((x-a)/(b-a), (c-x)/(c-b)), 0)

# Se adapto para el tp
def firing_strengths(x0_similarity, x0_rating, antecedentes_similarity, antecedentes_rating):
    '''
    Calcula los grados de activación para los valores de similitud y calificación dados, 
    basándose en sus respectivos antecedentes triangulares, y luego combina estos grados 
    usando el operador lógico AND (mínimo).

    Args:
        x0_similarity (float): Valor concreto y definido para la similitud.
        x0_rating (float): Valor concreto y definido para la calificación.
        antecedentes_similarity (list): Lista de números triangulares que serán antecedentes para la similitud.
        antecedentes_rating (list): Lista de números triangulares que serán antecedentes para la calificación.
        
    Returns:
        list: Lista combinada de los grados de activación para la similitud y la calificación,
              donde cada elemento es el mínimo de los grados de activación de similitud y calificación.

    Ejemplo:
        >>> firing_strengths(0.8, 4.5, [[0.6, 0.8, 1.0]], [[4, 4.5, 5]])
        [0.8]
    '''
    firing_strengths_similarity = [triangular_membership_degree(x0_similarity, antecedente) for antecedente in antecedentes_similarity]
    firing_strengths_rating = [triangular_membership_degree(x0_rating, antecedente) for antecedente in antecedentes_rating]
    
    # Combina las fuerzas de activación usando el operador AND (mínimo)
    combined_firing_strengths = [min(similarity, rating) for similarity, rating in zip(firing_strengths_similarity, firing_strengths_rating)]
    
    return combined_firing_strengths

def mamdani_inferencia(consecuentes, firing_strengths_list):
    '''
    Calcula la inferencia de mamdani, retornando la distribucion de pertenencia B'

    Args:
        consecuentes (list): lista con los numeros triangulares que seran consecuentes.
        firing_strengths_list (list): lista con los grados de activacion para x0
        
    Returns:
        list: La distribucion de pertenencia B'

    Example:
        >>> mamdani_inferencia([[2, 4, 6], [4, 6, 8]], [0.750000000000000, 0.250000000000000])
        [0.0,
        0.0030030030030030463,
        ...
        0.0030030030030028243,
        0]
        Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
    '''
    valores_y = range_values(consecuentes)
    b_prima = [max([min(alpha, triangular_membership_degree(y, consecuente)) for alpha, consecuente in zip(firing_strengths_list, consecuentes)]) for y in valores_y]
    return b_prima



def godel_inferencia(consecuentes, firing_strengths_list):
    '''
    Calcula la inferencia de godel, retornando la distribucion de pertenencia B'

    Args:
        consecuentes (list): lista con los numeros triangulares que seran consecuentes.
        firing_strengths (list): lista con los grados de activacion para x0
        
    Returns:
        list: la distribucion de pertenencia B'

    Example:
        >>> godel_inferencia([[2, 4, 6], [4, 6, 8]], [0.750000000000000, 0.250000000000000])
        [0.0,
        0.0030030030030030463,
        ...
        0.0030030030030028243,
        0]
        Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
    '''
    valores_y = range_values(consecuentes)
    
    b_prima_godel = []
    for y in valores_y:
        salida_godel = [triangular_membership_degree(y, consecuente) for consecuente in consecuentes]
        b_prima_godel.append(max(min(alpha, salida) for alpha, salida in zip(firing_strengths_list, salida_godel)))
    
    return b_prima_godel


def defuzzify(b_prima, consecuentes):
    '''
    Desdifusifica B' 

    Args:
        b_prima (list): la distribucion de pertenencia B'
        firing_strengths (list): lista con los grados de activacion para x0
        
    Returns:
        float: Valor desfusificado

    Example:
        >>> defuzzify(godel_inferencia([[2, 4, 6], [4, 6, 8]], [0.750000000000000, 0.250000000000000]), [[2, 4, 6], [4, 6, 8]]) 
        4.578946768883771
    '''
    valores_y = range_values(consecuentes)
    numerador = sum(y*b for y, b in zip(valores_y, b_prima))
    denominador = sum(b_prima)
    return numerador/denominador if denominador != 0 else 0

# Se modifico para el tp
def algoritmo(x0_similarity, x0_rating, antecedentes_similarity, antecedentes_rating, consecuentes, algoritmo_inferencia=mamdani_inferencia):
    '''
    Desdifusifica dos valores concretos y definidos, uno para la similitud y otro para la calificación, 
    bajo ciertos antecedentes y consecuentes utilizando un algoritmo de inferencia específico.

    Args:
        x0_similarity (float): Valor concreto y definido para la similitud.
        x0_rating (float): Valor concreto y definido para la calificación.
        antecedentes_similarity (list): Lista de números triangulares que serán antecedentes para la similitud.
        antecedentes_rating (list): Lista de números triangulares que serán antecedentes para la calificación.
        consecuentes (list): Lista con los números triangulares que serán consecuentes.
        algoritmo_inferencia (function): Puntero a función de inferencia, mamdani o godel por defecto. 
        
    Returns:
        tuple: Contiene dos elementos; el primero es una lista que representa la inferencia calculada,
               el segundo es un float que representa el valor desdifusificado.

    Ejemplo:
        >>> algoritmo(0.8, 4.5, [[0.6, 0.8, 1.0]], [[4, 4.5, 5]], [[3, 5, 7]], godel_inferencia)
        ([lista_inferencia], 5.25)
    '''
    #print(f'1- Input crisp Similarity Score:{x0_similarity} and Overall Product Rating:{x0_rating}')

    #print(f'2- Calculate firing strengths:')
    firing_strengths_list = firing_strengths(x0_similarity, x0_rating, antecedentes_similarity, antecedentes_rating)
    #print(firing_strengths_list)

    #print(f'3- Calculate inference')
    b_prima = algoritmo_inferencia(consecuentes, firing_strengths_list)
    #print(b_prima)

    #print(f'4- Defuzzify')
    y0 = defuzzify(b_prima, consecuentes)

    #print(f'5- Output: {y0}')

    return b_prima, y0