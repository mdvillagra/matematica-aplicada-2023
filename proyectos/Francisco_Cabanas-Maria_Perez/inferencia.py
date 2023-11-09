import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Antecedentes
rating = ctrl.Antecedent(np.arange(0, 6, 0.25), 'rating')
similarity = ctrl.Antecedent(np.arange(0, 11, 1), 'similarity')

# Consecuente
recommendation = ctrl.Consequent(np.arange(0, 3, 0.25), 'recommendation')

# Funciones de membresía
rating['poor'] = fuzz.trimf(rating.universe, [0, 0, 2])
rating['average'] = fuzz.trimf(rating.universe, [1, 2.5, 4])
rating['good'] = fuzz.trimf(rating.universe, [3, 5, 5])
# rating.view()
# input("Figura Rating")

similarity['poor'] = fuzz.trimf(similarity.universe, [0, 0, 4])
similarity['average'] = fuzz.trimf(similarity.universe, [2, 5, 8])
similarity['good'] = fuzz.trimf(similarity.universe, [6, 10, 10])
# similarity.view()
# input("Figura Similarity")

recommendation['not_recommend'] = fuzz.trimf(recommendation.universe, [0, 0, 0.75])
recommendation['likely_recommend'] = fuzz.trimf(recommendation.universe, [0.2, 0.95, 1.7])
recommendation['recommend'] = fuzz.trimf(recommendation.universe, [1.5, 2, 2])
# recommendation.view()
# input("Figura Recommendation")

# Reglas
rule1 = ctrl.Rule(rating['poor'] & similarity['poor'], recommendation['not_recommend'])
rule2 = ctrl.Rule(rating['poor'] & similarity['average'], recommendation['not_recommend'])
rule3 = ctrl.Rule(rating['poor'] & similarity['good'], recommendation['likely_recommend'])
rule4 = ctrl.Rule(rating['average'] & similarity['poor'], recommendation['not_recommend'])
rule5 = ctrl.Rule(rating['average'] & similarity['average'], recommendation['likely_recommend'])
rule6 = ctrl.Rule(rating['average'] & similarity['good'], recommendation['recommend'])
rule7 = ctrl.Rule(rating['good'] & similarity['poor'], recommendation['likely_recommend'])
rule8 = ctrl.Rule(rating['good'] & similarity['average'], recommendation['likely_recommend'])
rule9 = ctrl.Rule(rating['good'] & similarity['good'], recommendation['recommend'])

# Sistema de control
sistema_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
sistema_inferencia = ctrl.ControlSystemSimulation(sistema_control)

# # Entradas
# sistema_inferencia.input['rating'] = 5
# sistema_inferencia.input['similarity'] = 10

# # Computar
# sistema_inferencia.compute()

# # Salida
# print(sistema_inferencia.output['recommendation'])
