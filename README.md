# Análisis de Sentimientos con SVM

### Introducción

Este proyecto está diseñado para facilitar el análisis de sentimientos de reseñas mediante el uso de Support Vector Machines (SVM). El sistema procesa datasets de reseñas, clasifica las opiniones en categorías de sentimiento (positivas, neutrales y negativas) y entrena un modelo SVM optimizado para realizar predicciones sobre nuevos datos.

El código está dividido en varios scripts para modularizar el proceso y permitir la reutilización. Cada script tiene una funcionalidad específica, desde la preparación de los datos hasta el entrenamiento y evaluación del modelo.

---

### Descripción de los Scripts

### **1. `process-dataset-train.py`**

Este script se encarga de procesar y combinar los datasets utilizados para entrenar el modelo SVM. 

#### **Funciones principales**
- **`load_dataset(path)`**: Carga un archivo CSV como un DataFrame de pandas.
- **`categorize_sentiment(target)`**: Clasifica una puntuación numérica en una categoría de sentimiento (`negative`, `neutral`, `positive`, `unknown`).

#### **Flujo del script**
1. Define las rutas base y específicas para los datasets.
2. Carga y renombra las columnas relevantes (`Target` y `Text`).
3. Combina los datasets en un único DataFrame.
4. Crea una nueva columna `Sentiment` basada en los valores de `Target`.
5. Guarda el dataset procesado en un archivo CSV.

#### **Salida**
Un archivo CSV combinado listo para ser utilizado en el entrenamiento del modelo.

---

### **2. `process-dataset-test.py`**

Este script realiza un procesamiento similar al de `process-dataset-train.py`, pero está enfocado en los datasets de prueba, que se utilizarán para evaluar el modelo SVM.

#### **Funciones principales**
- **`load_dataset(path)`** y **`categorize_sentiment(target)`**: Mismas funcionalidades que en el script de entrenamiento.

#### **Flujo del script**
1. Define las rutas base y específicas para los datasets de prueba.
2. Procesa y combina los datos en un único DataFrame.
3. Clasifica los puntajes de `Target` en categorías de sentimiento.
4. Exporta los datos procesados a un archivo CSV.

#### **Salida**
Un archivo CSV con datos de prueba procesados.

---

### **3. `train_svm.py`**

Este script es el núcleo del proyecto, ya que entrena el modelo SVM para análisis de sentimientos.

#### **Funciones principales**
- **`setup_nltk()`**: Descarga los recursos necesarios para la tokenización y lematización.
- **`initialize_stopwords()`**: Configura las palabras de parada.
- **`preprocess_tweet(tweet, stop_words, lemmatizer)`**: Limpia y preprocesa el texto.
- **`balance_dataset()`**: Equilibra las clases del dataset para evitar sesgos.
- **`vectorize_data()`**: Transforma el texto en características numéricas usando TF-IDF.
- **`split_dataset()`**: Divide los datos en conjuntos de entrenamiento y prueba.
- **`optimize_svm()`**: Optimiza los hiperparámetros del modelo SVM usando GridSearchCV.

#### **Flujo del script**
1. Procesa el dataset cargado, limpiando el texto y clasificando sentimientos.
2. Vectoriza el texto utilizando un TF-IDF Vectorizer.
3. Divide el dataset en datos de entrenamiento y prueba.
4. Optimiza los parámetros del modelo SVM y lo entrena con los datos de entrenamiento.
5. Evalúa el modelo y genera gráficos de análisis (matriz de confusión, reporte de clasificación y barra de precisión).
6. Guarda el modelo SVM y el vectorizador en archivos `.pkl` para uso futuro.

#### **Salida**
- Un modelo SVM entrenado y guardado.
- Análisis visual del rendimiento del modelo.

---

### **4. `test.py`**

Este script permite realizar pruebas y predicciones usando el modelo SVM entrenado, generando resultados con los datos de entrada.

#### **Funciones principales**
- **`load_model(model_path)`**: Carga el modelo SVM previamente entrenado desde un archivo `.pkl`.
- **`load_vectorizer(vectorizer_path)`**: Carga el vectorizador TF-IDF.
- **`preprocess_input(input_text, stop_words, lemmatizer)`**: Preprocesa el texto de entrada antes de la predicción.
- **`predict_sentiment(model, vectorizer, input_text)`**: Realiza una predicción de sentimiento en el texto de entrada.

#### **Flujo del script**
1. Carga el modelo SVM y el vectorizador guardados.
2. Preprocesa el texto de entrada, limpiando y transformando el texto.
3. Realiza la predicción utilizando el modelo entrenado.
4. Muestra el resultado de la predicción (sentimiento).

#### **Salida**
- El sentimiento predicho para el texto de entrada.

---

### Requisitos

- **Python 3.8+**
- **Bibliotecas necesarias**:
  - `pandas`
  - `numpy`
  - `sklearn`
  - `matplotlib`
  - `seaborn`
  - `nltk`
  - `joblib`
  - `kagglehub`

---

### Instrucciones de Uso

1. **Preparación del entorno**
   - Instala las bibliotecas requeridas utilizando:
     ```bash
     pip install -r requirements.txt
     ```
   - Configura las rutas base en los scripts para tus datasets.

2. **Procesamiento de datos**
   - Ejecuta `process-dataset-train.py` para preparar el dataset de entrenamiento.
   - Ejecuta `process-dataset-test.py` para preparar el dataset de prueba.

3. **Entrenamiento del modelo**
   - Ejecuta `train_svm.py` para entrenar y evaluar el modelo SVM.

4. **Predicciones**
   - Ejecuta `test.py` para hacer predicciones de sentimiento con el modelo entrenado.

---

### Salidas Esperadas

- **Archivos procesados**: 
  - `all_datasets_reviews.csv` (entrenamiento).
  - `iphone_reviews_processed.csv` (test final).
- **Modelo y vectorizador**: 
  - `svm_model.pkl`.
  - `tfidf_vectorizer.pkl`.
- **Gráficos de análisis**:
  - Matriz de confusión.
  - Reporte de clasificación.
  - Gráfico de precisión.

---

### Autor

Este proyecto fue desarrollado por **Udane Viñas Templado** como parte de un proyecto para análisis de sentimientos con aprendizaje automático.
