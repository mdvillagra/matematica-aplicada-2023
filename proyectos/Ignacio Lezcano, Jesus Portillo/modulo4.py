import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

def definir_reglas_difusas():
    """
    Define un sistema de control difuso con variables de entrada y salida, y reglas difusas.

    Returns:
    - ctrl.ControlSystemSimulation: Objeto de simulación del sistema de control difuso.
    """
    # Crear las variables de entrada y salida difusa
    similarity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'similarity')
    rating = ctrl.Antecedent(np.arange(0, 5.01, 0.01), 'rating')
    recomendacion = ctrl.Consequent(np.arange(0, 1, 0.1), 'recomendacion')

    # Definir funciones de membresía para las variables
    similarity.automf(names=['pobre', 'promedio', 'bueno', 'excelente'])
    rating.automf(names=['pobre', 'promedio', 'bueno', 'excelente'])
    recomendacion.automf(names=['No Recomendado', 'Posiblemente Recomendado', 'Recomendado', 'Altamente Recomendado'])

    # Reglas para la variable de salida "recomendacion"
    reglas = [
        ctrl.Rule(similarity['excelente'] & rating['excelente'], recomendacion['Altamente Recomendado']),
        ctrl.Rule(similarity['excelente'] & rating['bueno'], recomendacion['Altamente Recomendado']),
        ctrl.Rule(similarity['promedio'] & rating['excelente'], recomendacion['Altamente Recomendado']),
        ctrl.Rule(similarity['bueno'] & rating['excelente'], recomendacion['Altamente Recomendado']),

        ctrl.Rule(similarity['pobre'] & rating['excelente'], recomendacion['Recomendado']),
        ctrl.Rule(similarity['promedio'] & rating['bueno'], recomendacion['Recomendado']),
        ctrl.Rule(similarity['bueno'] & rating['bueno'], recomendacion['Recomendado']),
        ctrl.Rule(similarity['excelente'] & rating['promedio'], recomendacion['Recomendado']),

        ctrl.Rule(similarity['bueno'] & rating['promedio'], recomendacion['Posiblemente Recomendado']),
        ctrl.Rule(similarity['excelente'] & rating['pobre'], recomendacion['Posiblemente Recomendado']),
        ctrl.Rule(similarity['promedio'] & rating['promedio'], recomendacion['Posiblemente Recomendado']),        
        ctrl.Rule(similarity['pobre'] & rating['bueno'], recomendacion['Posiblemente Recomendado']),
        
        ctrl.Rule(similarity['pobre'] & rating['pobre'], recomendacion['No Recomendado']),
        ctrl.Rule(similarity['pobre'] & rating['promedio'], recomendacion['No Recomendado']),
        ctrl.Rule(similarity['promedio'] & rating['pobre'], recomendacion['No Recomendado']),  
        ctrl.Rule(similarity['bueno'] & rating['pobre'], recomendacion['No Recomendado']), 

    ]

    # Crear el sistema de control difuso
    sistema_control = ctrl.ControlSystem(reglas)
    return ctrl.ControlSystemSimulation(sistema_control)

def aplicar_inferencia_difusa(similitud, rating, sistema):
    """
    Aplica inferencia difusa al sistema con valores de entrada dados y devuelve el resultado.

    Args:
    - similitud (float): Valor de similitud entre productos.
    - rating (float): Valor de rating del producto.
    - sistema (ctrl.ControlSystemSimulation): Objeto de simulación del sistema de control difuso.

    Returns:
    - float: Valor de recomendación difusa.
    """
    # Simular el sistema con valores de entrada
    sistema.input['similarity'] = similitud
    sistema.input['rating'] = rating
    # Obtener el resultado
    sistema.compute()
    return sistema.output['recomendacion']
