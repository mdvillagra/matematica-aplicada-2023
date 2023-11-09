
# Sistema de recomendación difuso 

Un sistema de recomendación difuso para predecir los intereses de los clientes basado en el articulo

**Karthik, R. V., & Ganapathy, S. (2021). A fuzzy recommendation system for predicting the customers interests using sentiment analysis and ontology in e-commerce. Applied Soft Computing, 108, 107396.** [Link](https://drive.google.com/file/d/1gc8KolYckROJC7VXRo2kedTLMeYouXV-/view?usp=sharing)

con 5 modulos implementados del mismo:

* **Módulo 1:** Lector de datasets
* **Módulo 2:** Rating score
* **Módulo 3:** Similarity score
* **Módulo 4:** Sistema de inferencia difuso
* **Módulo 5:** Defusificador



## Requisitos

 - Lenguaje de programacion: [Python](https://www.python.org/downloads/)
 - Datasets - Complete review data: [Amazon Review Data (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
 - Librerias: Json, Gzip, Pandas, Numpy, Matplotlib y Sklearn
 - Compiladores(opcional): Visual Studio Code, Pycharm, otros compatibles.

## Instrucciones

Instalar tanto el lenguaje de programacion como dichas librerias,
lo puedes hacer desde de la terminal usando:

Verificar python
```bash
  Python --version
```

Librerias
```bash
  pip install nombre_libreria_correspondiente
```

Tambien debes instalar el duo de datasets de "Magazine Subscriptions"

* [Review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Magazine_Subscriptions.json.gz)
* [Metadata](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Magazine_Subscriptions.json.gz)

Puedes usar cualquier otro, en ese caso tendras que cambiar las rutas de la siguiente linea del codigo fuente junto con el ID_Usuario sacado de reviewerID dentro del archivo Review (Nombre_Archivo.json.gz) de tu duo seleccionado

```bash
  FinalDataFrame, graphics = fuzzy_recommendation_system('Nombre_Archivo.json.gz',  
                                                    'meta_Nombre_Archivo.json.gz', 'ID_Usuario')
```
Una vez todo instalado te facilitaria tener los archivos dentro de una misma carpeta, caso contrario colocar ruta correcta en cada parametro de la linea del codigo anterior mostrada

**Contenido de la carpeta**
* proyecto.py (codigo fuente)
* Magazine_Subscriptions.json.gz (u otro)
* meta_Magazine_Subscriptions.json.gz (u otro) 

Abres la carpeta con el compilador de tu eleccion y ejecutas el codigo, otra opcion seria hacerlo desde la terminal situandote en la carpeta(ruta donde se encuentra el codigo fuente junto con los datasets) y escribiendo:
```bash
  python proyecto.py
```
Resultado del ejemplo:
```bash
  User name:  John L. Mehlmauer

The final results for (reviewerID:  A5QQOOZJOVPSF )  John L. Mehlmauer  are:
            similarity_score  overall_pr               result
B000FTJ7BE          0.707107    4.696629     Highly recommend
B001GDJ4OS          0.816497    4.404762     Highly recommend
B01F2MKW0I          0.816497    4.597826     Highly recommend
B000PUAI3E          0.707107    4.408451     Highly recommend
B00005N7SD          0.707107    3.795181            Recommend
B0058EONOM          0.816497    4.600000     Highly recommend
B00ATQ6FPY          0.912871    4.386364     Highly recommend
B00FP59VGY          0.707107    4.900000     Highly recommend
B00007AZRH          0.912871    4.535088     Highly recommend
B000063XJL          0.912871    3.996114     Highly recommend
B00007AZWJ          0.912871    4.073801     Highly recommend
B00005N7R5          0.912871    4.311594     Highly recommend
B00006KNXP          0.577350    4.535714     Highly recommend
B00005NIO6          0.408248    4.350000            Recommend
B00006K3EU          0.707107    4.521739     Highly recommend
B00005N7RQ          0.577350    4.200000     Highly recommend
B00006J9HW          0.707107    4.260606     Highly recommend
B00006KFT2          0.577350    4.508475     Highly recommend
B00006LK8F          0.408248    4.266667            Recommend
B00VQTC94E          0.408248    4.833333            Recommend
B00007AVYH          0.577350    4.100000     Highly recommend
B00009MQ5Q          0.408248    4.590909            Recommend
B00007B2N1          0.408248    4.436364            Recommend
B00A6IMSTC          0.577350    3.800000            Recommend
B00006KP86          0.577350    3.333333            Recommend
B00006KVLZ          0.577350    4.545455     Highly recommend
B0006PUYLY          0.408248    4.178218            Recommend
B001684M22          0.408248    4.155172            Recommend
B000063XJR          0.408248    4.526316            Recommend
B003K195Y8          0.408248    3.777778  Likely to recommend
B00005NIRE          0.408248    3.809524  Likely to recommend
B000O1PKOG          0.408248    4.417808            Recommend
B00ZDXLBEI          0.408248    4.250000            Recommend
B00006KPU2          0.577350    4.208333     Highly recommend
B0000TL75I          0.577350    5.000000     Highly recommend
B015GEA15S          0.577350    4.500000     Highly recommend
B00AVCRAB4          0.577350    5.000000     Highly recommend
B00005N7VQ          0.408248    3.992188            Recommend
B00006KDW3          0.408248    4.083333            Recommend
B00006KH16          0.707107    3.818182            Recommend
B00005N7VP          0.408248    3.759124  Likely to recommend
B00006KTZP          0.408248    5.000000            Recommend
B0000A0O0G          0.408248    3.900000            Recommend
B00005N7OF          0.408248    4.360000            Recommend
B000AMXXF2          0.408248    2.750000  Likely to recommend
```
## Notas

El tiempo de ejecucion puede variar dependiendo de los tamanos de los datasets utilizados, para este ejemplo el tiempo de ejecucion es de aproximadamente 2 minutos

**Recomendacion:** no utilizar datasets muy grandes



## Autor/a

[@Abinues](https://github.com/Abinues)


