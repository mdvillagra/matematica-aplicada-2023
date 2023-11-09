# Readme para Proyecto de Recomendaciones de Productos

## Descripción

Este proyecto implementa un sistema de recomendaciones de productos utilizando análisis de opiniones de reseñas y técnicas de procesamiento de lenguaje natural (NLP). Calcula las puntuaciones de similitud entre productos y proporciona recomendaciones personalizadas.

## Requisitos Previos

Antes de comenzar, asegúrate de tener Python instalado en tu sistema. Este proyecto utiliza Pipenv para la gestión de dependencias y entornos virtuales.

## Instalación

Sigue estos pasos para configurar el entorno de desarrollo y ejecutar el proyecto:

### 1. Clonar el Repositorio

Primero, clona el repositorio del proyecto a tu máquina local usando:

```bash
git clone [URL del Repositorio]
cd [Nombre del Directorio del Repositorio]
```

### 2. Instalar Pipenv

Si no tienes Pipenv instalado, puedes instalarlo usando pip. Pipenv te ayuda a gestionar dependencias y entornos virtuales. Ejecuta el siguiente comando para instalarlo:

```bash
pip install pipenv
```

### 3. Configurar el Entorno Virtual

Una vez instalado Pipenv, puedes configurar un entorno virtual y instalar las dependencias necesarias. En el directorio del proyecto, ejecuta:

```bash
pipenv install
```

Este comando creará un entorno virtual para el proyecto y instalará todas las dependencias especificadas en el archivo `Pipfile`.

### 4. Activar el Entorno Virtual

Para activar el entorno virtual y trabajar dentro de él, utiliza:

```bash
pipenv shell
```

### 5. Ejecutar el Proyecto

Ahora que tienes el entorno activado y las dependencias instaladas, puedes ejecutar el script principal del proyecto:

```bash
cd src
python main.py
```

## Uso

El script `main.py` procesará los datos, calculará las puntuaciones de similitud entre productos y generará recomendaciones, que se guardarán en un archivo JSON (`product_recommendations.json`).

