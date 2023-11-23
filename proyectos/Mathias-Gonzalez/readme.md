# Instrucciones de ejecucion
El programa fue desarrollado en python 3.10.12 y se ejecuta con el siguiente codigo:
```
python main.py -d ruta_dataset_de_reviews -m ruta_dataset_de_metadatos
``` 
Para el dataset de reviews se aceptan archivos .json.gz y .csv
Para el dataset de metadatos se aceptan archivos .json.gz

Se utiliza un solo una muestra pequeña aleatoria del dataset de reviews para acelerar la ejecucion.
Ademas el sistema elige un usuario aleatorio para simular la solicitud de recomendacion de ese usuario.

## Dependencias
Se pueden instalar las dependencias ejecutando ```pip install -r dependencias.txt```
- numpy           1.26.1
- pandas          2.1.2
- scikit-learn    1.3.2