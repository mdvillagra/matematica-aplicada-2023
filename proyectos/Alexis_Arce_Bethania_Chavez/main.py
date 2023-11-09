from modulos.lector_datasets import getDF # modulo 1 lector de datasets
from modulos.rating_score import calcular_rating_score # modulo 2 rating score
from modulos.similarity_score import similarity_score_total#, similarity_score_un_item  # modulo 3 similarity score
from modulos.sistema_inferencia_difusa import sistema_inferencia_difusa # modulo 4 sistema de inferencia difusa
from modulos.defusicador import defusicador # modulo 5 desuficador
from IPython.display import display     

print("\n########################################################################################################################################")

print("  __________________________________")
print(" | Modulo 1 Lector de datasets      |")
print(" |__________________________________|\n")

df = getDF('./data/Magazine_Subscriptions_5.json.gz')
display(df)
print("\n########################################################################################################################################")

print("  __________________________________")
print(" | Modulo 2 Rating Score            |")
print(" |__________________________________|\n")
rating_score = calcular_rating_score(df)

print("\n########################################################################################################################################")

print("  __________________________________")
print(" | Modulo 3 Similarity Score        |")
print(" |__________________________________|\n")
# Calcula la similaridad a partir de un item: similarity_score_un_item(dataframe, indice_del_item)
#print("Calculo de similaridad score a partir de un item: \n")
#similarity_score_un_item(df,0)

print("Calculo de similaridad score total: \n")
# Calcular la similaridad de todos los items
similarity_score = similarity_score_total(df)

print("\n########################################################################################################################################")

print("  _______________________________________")
print(" | Modulo 4 Sistema de inferencia difuso |")
print(" |_______________________________________|\n")

sistema_inferencia_difusa(similarity_score, rating_score)

print("\n########################################################################################################################################")
print("  _______________________________________")
print(" | Modulo 5 Defusicador                  |")
print(" |_______________________________________|\n")
defusicador(similarity_score, rating_score)
print("########################################################################################################################################")

