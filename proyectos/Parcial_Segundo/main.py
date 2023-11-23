import skfuzzy as fuzz
from skfuzzy import control as ctrl
from modulo1 import cargar_datos
from modulo2 import similitud_entre_conjuntos
from modulo3 import obtener_promedio_rating_por_producto
from modulo4 import aplicar_inferencia_difusa, definir_reglas_difusas
from modulo5 import defusificar

# Menú principal
def menu():
    """
    Despliega un menú para que el usuario seleccione una opción.

    Returns:
    - opcion (str): Opción seleccionada por el usuario.
    """
    print("\nPor favor seleccione una opción:")
    print("1. Ingresar ID del reviewer del cual se desea conocer la lista de recomendación.")
    print("2. Terminar Programa.")
    opcion = input("Opción: ")
    return opcion

# Módulo adicional: Recomendación de productos
def recomendar(reviewer, datos_reviews, datos_meta, rating_promedio_por_producto):
    """
    Genera recomendaciones de productos para un reviewer dado.

    Args:
    - reviewer (str): ID del reviewer.
    - datos_reviews (list): Lista de datos de reviews.
    - datos_meta (list): Lista de datos de productos.
    - rating_promedio_por_producto (dict): Diccionario de promedio de rating por producto.

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
        print(f"El reviewer {reviewer} no ha revisado ningún producto.")



def generar_recomendaciones(producto_principal, datos_meta, rating_promedio_por_producto):
    """
    Genera recomendaciones de productos similares para un producto dado.

    Args:
    - producto_principal (str): ID del producto principal.
    - datos_meta (list): Lista de datos de productos.
    - rating_promedio_por_producto (dict): Diccionario de promedio de rating por producto.

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
                print(f"Resultado de inferencia: {valor_de_inferencia}")

                valor_de_inferencia = aplicar_inferencia_difusa(similaridad, rating_producto_secundario, definir_reglas_difusas())
                
                # Defusificar y obtener la recomendación
                nivel_de_recomendacion = defusificar(valor_de_inferencia)

                # Imprimir resultados
                if nivel_de_recomendacion != "No Recomendado":
                    print(f"Asin: {producto['asin']}, Producto: {producto['title']}, Recomendación: {nivel_de_recomendacion}\n")
                else:
                    print("No recomendado\t")


if __name__ == "__main__":
    # Módulo 1: Cargar datos
    datos_reviews = cargar_datos('Software.json')
    datos_meta = cargar_datos('meta_Software.json')

    # Módulo 2: Similarity Score
    # (No se necesita implementar aquí, ya que se calcula dinámicamente al recomendar productos)

    # Módulo 3: Rating Score
    rating_promedio_por_producto = obtener_promedio_rating_por_producto(datos_reviews)

    Product_por_reviewer = {}
    
    # Menú principal
    while True:
        opcion = menu()
        if opcion == "1":
            reviewer = input("Ingrese ID: ")
            recomendar(reviewer, datos_reviews, datos_meta, rating_promedio_por_producto)
        elif opcion == "2":
            break
        else:
            print("Opción no válida. Por favor, seleccione una opción válida.")
