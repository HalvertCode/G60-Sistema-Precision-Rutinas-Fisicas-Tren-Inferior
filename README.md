# 🏋️‍♂️ G60: Sistema de Precisión de Rutinas Físicas - Tren Inferior

Este proyecto implementa un sistema de evaluación y corrección de rutinas físicas enfocadas en el tren inferior, utilizando detección de pose humana para mejorar la precisión y seguridad durante la ejecución de ejercicios.

## 🎯 Objetivo

Desarrollar una herramienta que permita analizar y corregir la técnica de ejercicios como sentadillas, desplantes y elevaciones de talones, proporcionando retroalimentación en tiempo real al usuario.

## 🧰 Tecnologías Utilizadas

- **Python**: Lenguaje principal del proyecto.
- **OpenCV**: Procesamiento de imágenes y video.
- **MediaPipe**: Detección de poses humanas.
- **scikit-learn**: Entrenamiento de modelos de clasificación.
- **NumPy** y **Pandas**: Manipulación y análisis de datos.

## 📁 Estructura del Proyecto

- `collect_*.py`: Scripts para la recopilación de datos de cada ejercicio.
- `*_form_checker.py`: Módulos que evalúan la forma del ejercicio.
- `*_app.py`: Interfaces para la interacción con el usuario.
- `train_*.py`: Scripts para entrenar modelos de clasificación.
- `*.pkl`: Modelos entrenados almacenados en formato pickle.
- `*.csv`: Conjuntos de datos recopilados.
- `requirements.txt`: Lista de dependencias del proyecto.

## 🚀 Cómo Empezar

1. Clona el repositorio:

   ```bash
   git clone https://github.com/HalvertCode/G60-Sistema-Precision-Rutinas-Fisicas-Tren-Inferior.git
   cd G60-Sistema-Precision-Rutinas-Fisicas-Tren-Inferior
   ```

2. Crea un entorno virtual y actívalo:

   ```bash
   python -m venv env
   source env/bin/activate  # En Windows: env\Scripts\activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Ejecuta la aplicación deseada:

   ```bash
   python squats_app.py
   ```

## 📦 Archivos de Datos

Los archivos de datos (.pkl y .csv) del repositorio se pueden descargar desde [este enlace de Google Drive](https://drive.google.com/drive/folders/1azSR81gyLWGCcZueUkVx534DCR30gwWR?usp=sharing).

## 📹 Presentación del Proyecto

Para una demostración visual del funcionamiento del sistema, puedes ver el siguiente video:

[![Presentación G60](https://img.youtube.com/vi/N3otuIkYhJc/0.jpg)](https://www.youtube.com/watch?v=N3otuIkYhJc)

## 👥 Autores

- **Halvert Alessandro Guerrero Puicón**
- **Jose Jean Pierre Suclupe Vela**