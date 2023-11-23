import skfuzzy as fuzz
from skfuzzy import control as ctrl
from modulo1 import cargar_datos
from modulo2 import similitud_entre_conjuntos
from modulo3 import obtener_promedio_rating_por_producto
from modulo4 import aplicar_inferencia_difusa, definir_reglas_difusas

# Módulo 5: Defusificador
def defusificar(valor_de_inferencia):
    """
    Defusifica el valor de inferencia para obtener un nivel de recomendación.

    Args:
    - valor_de_inferencia (float): Valor resultante de la inferencia difusa.

    Returns:
    - str: Nivel de recomendación ("No Recomendado", "Posiblemente Recomendado", "Recomendado", "Altamente Recomendado").
    """
    if valor_de_inferencia <= 0.15:
        nivel_de_recomendacion = "No Recomendado"
    elif valor_de_inferencia <= 0.25:
        nivel_de_recomendacion = "Posiblemente Recomendado"
    elif valor_de_inferencia <= 0.4:
        nivel_de_recomendacion = "Recomendado"
    else:
        nivel_de_recomendacion = "Altamente Recomendado"
    return nivel_de_recomendacion

# Menú principal
def menu():
    """
    Despliega un menú para que el usuario seleccione una opción.

    Returns:
    - str: Opción seleccionada por el usuario.
    """
    print("\nPor favor, seleccione una opción:")
    print("1. Ingresar ID del reviewer del cual se desea conocer la lista de recomendación.")
    print("2. Terminar Programa.")
    opcion = input("Opción: ")
    return opcion

# Módulo adicional: Recomendación de productos
def recomendar(reviewer, datos_reviews, datos_meta, rating_promedio_por_producto):
    """
    Genera recomendaciones de productos para un revisor específico.

    Args:
    - reviewer (str): ID del revisor.
    - datos_reviews (list): Lista de objetos JSON con datos de revisiones.
    - datos_meta (list): Lista de objetos JSON con datos de productos.
    - rating_promedio_por_producto (dict): Diccionario con el promedio de rating por producto.

    Returns:
    - None
    """
    
    global Product_por_reviewer  # Agrega esta línea para indicar que se utilizará la variable global

    # Actualizar la variable global después de la recomendación
    Product_por_reviewer[reviewer] = [review['asin'] for review in datos_reviews if review['reviewerID'] == reviewer]

    if reviewer in Product_por_reviewer:
        productos_revisados = Product_por_reviewer[reviewer]
        cont2 = 0
        for producto in productos_revisados:
            cont2 += 1
            if cont2 > 1000:
                break
            generar_recomendaciones(producto, datos_meta, rating_promedio_por_producto)
    else:
        print(f"No se encuentran revisiones por parte del reviewer {reviewer}.")


def generar_recomendaciones(producto_principal, datos_meta, rating_promedio_por_producto):
    """
    Genera recomendaciones de productos en función de un producto principal.

    Args:
    - producto_principal (str): ID del producto principal.
    - datos_meta (list): Lista de objetos JSON con datos de productos.
    - rating_promedio_por_producto (dict): Diccionario con el promedio de rating por producto.

    Returns:
    - None
    """

    cont = 0
    for dato in datos_meta:
        if dato['asin'] == producto_principal:
            producto = dato
            break
    conjunto_A_productos = set()

    # Verificar si 'also_buy' existe y no es None antes de actualizar el conjunto
    if producto.get('also_buy'):
        conjunto_A_productos.update(producto['also_buy'])

    # Verificar si 'also_view' existe y no es None antes de actualizar el conjunto
    if producto.get('also_view'):
        conjunto_A_productos.update(producto['also_view'])

    for producto_secundario in datos_meta:
        cont += 1
        if cont > 1000:
            break 
        if producto_secundario['asin'] != producto_principal:
            conjunto_B_productos = set()

            # Verificar si 'also_buy' existe y no es None antes de actualizar el conjunto
            if producto_secundario.get('also_buy'):
                conjunto_B_productos.update(producto_secundario['also_buy'])

            # Verificar si 'also_view' existe y no es None antes de actualizar el conjunto
            if producto_secundario.get('also_view'):
                conjunto_B_productos.update(producto_secundario['also_view'])

                similaridad = similitud_entre_conjuntos(producto, producto_secundario)
                rating_producto_secundario = 0 if producto_secundario['asin'] not in rating_promedio_por_producto else rating_promedio_por_producto[producto_secundario['asin']]['promedio_rating']

                # Aplicar inferencia difusa
                valor_de_inferencia = aplicar_inferencia_difusa(similaridad, rating_producto_secundario, definir_reglas_difusas())

                # Imprimir resultado de inferencia
                # print(f"Resultado de inferencia: {valor_de_inferencia}")

                # Defusificar y obtener la recomendación
                nivel_de_recomendacion = defusificar(valor_de_inferencia)

                # Imprimir resultados
                if nivel_de_recomendacion != "No Recomendado":
                    print(f"Asin: {producto['asin']}, Producto: {producto['title']}, Recomendación: {nivel_de_recomendacion}\n")


if __name__ == "__main__":
    # Módulo 1: Cargar datos
    datos_reviews = cargar_datos('Software.json')
    datos_meta = cargar_datos('meta_Software.json')

    # Módulo 2: Similarity Score
    # (No se necesita implementar aquí, ya que se calcula dinámicamente al recomendar productos)

    # Módulo 3: Rating Score
    rating_promedio_por_producto = obtener_promedio_rating_por_producto(datos_reviews)

    Product_por_reviewer = {}
