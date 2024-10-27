# fuzzyX - Sentiment Analysis + Sentiment140 + Python

![Python](https://img.shields.io/badge/Python-3.12.5-3776AB?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-150458?style=for-the-badge&logo=pandas)
![NLTK](https://img.shields.io/badge/NTLK-3.8.1-107C10?style=for-the-badge&logo=nltk)
![Fuzzy](https://img.shields.io/badge/Skfuzzy-0.4.2-3E8E41?style=for-the-badge&logo=fuzzylogic)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.2-239120?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

fuzzyX es un proyecto de análisis de sentimientos que combina datos del conjunto *Sentiment140* con Python, utilizando lógica difusa para el análisis y clasificación de emociones. El proyecto implementa varios módulos basados en el siguiente artículo:

**Vashishtha, S., & Susan, S. (2019). Fuzzy rule based unsupervised sentiment analysis from social media posts. Expert Systems with Applications, 138, 112834.**  
[Enlace al artículo](https://www.researchgate.net/profile/Srishti-Vashishtha-2/publication/334622166_Fuzzy_Rule_based_Unsupervised_Sentiment_Analysis_from_Social_Media_Posts/links/5ece42174585152945149e5b/Fuzzy-Rule-based-Unsupervised-Sentiment-Analysis-from-Social-Media-Posts.pdf)

## 🚀 Características principales
 - Lógica difusa: Se emplea skfuzzy para mejorar la clasificación mediante reglas difusas, proporcionando un análisis más robusto de los sentimientos.
 - Dataset: Los datos provienen de Sentiment140 y son procesados mediante Pandas para facilitar su manipulación.
 - Análisis basado en reglas: Se basa en el artículo Vashishtha, S., & Susan, S. (2019), donde se especifica el uso de reglas difusas para analizar sentimientos en redes sociales.

## 🛠️ Tecnologías usadas

- **Lenguaje principal**: Python 3.12.5
- **Librerías principales**: Pandas, NLTK, Skfuzzy, Matplotlib
- **Análisis de sentimientos**: NLTK SentimentIntensityAnalyzer
- **Visualización**: Matplotlib
- **Lógica difusa**: Skfuzzy

## 📦 Dependencias principales

Este proyecto utiliza las siguientes dependencias:

```bash
pandas==2.0.3
re==2.2.1
numpy==1.26.0
skfuzzy==0.4.2
matplotlib==3.7.2
nltk==3.8.1
tabulate==0.9.0
time==3.4.0
os==2.1.0
```

## 🔧 Instalación
Sigue los siguientes pasos para instalar el proyecto en tu entorno local.

1. Clonar el repositorio
```bash
git clone https://github.com/kevinfpuna/fuzzyX.git
cd fuzzyX
```
2. Crear un entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
```
o
```bash
python -m venv venv
venv\Scripts\activate  # En Windows
```
3. Instalar las dependencias
Instala todas las dependencias del proyecto con:
```bash
pip install -r requirements.txt
```

4. Ejecutar el análisis de sentimientos
Puedes iniciar el análisis ejecutando el script principal:
```bash
python main.py
```
## 📄 Licencia
Este proyecto está bajo la licencia MIT. Puedes ver más detalles en el archivo LICENSE.

## 👥 Integrantes del proyecto:

Kevin Galeano

Majo Duarte