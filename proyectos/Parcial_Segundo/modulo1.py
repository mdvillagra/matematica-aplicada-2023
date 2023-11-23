import json

# Módulo 1: Lector de datasets
def cargar_datos(ruta_archivo):
    """
    Lee un archivo JSON línea por línea y carga los datos en una lista.

    Args:
    - ruta_archivo (str): Ruta del archivo JSON.

    Returns:
    - list: Lista de objetos JSON cargados desde el archivo.
    """
    datos = []
    with open(ruta_archivo, 'r') as archivo:
        for linea in archivo:
            objeto_json = json.loads(linea)
            datos.append(objeto_json)
    return datos

Product_por_reviewer = {} # Diccionario vacio para rastrear productos por reviewer