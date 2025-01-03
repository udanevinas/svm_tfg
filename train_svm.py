import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuración inicial y descargas necesarias para NLTK
def setup_nltk():
    """
    Descarga los recursos necesarios para trabajar con NLTK: stopwords, tokenizer y lematizador.

    Esta función no toma ni retorna parámetros.
    """
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

# Inicializa la lista de stopwords
def initialize_stopwords():
    """
    Inicializa un conjunto de stopwords en inglés y agrega términos comunes encontrados en tweets.

    Returns:
        set: Conjunto de stopwords.
    """
    stop_words = set(stopwords.words('english'))
    stop_words.update(['rt', 'via', 'us', 'u'])
    return stop_words

# Carga un dataset desde un archivo CSV
def load_dataset(path):
    """
    Carga un dataset desde un archivo CSV.

    Args:
        path (str): Ruta del archivo CSV.

    Returns:
        pandas.DataFrame: Dataset cargado como DataFrame.
    """
    return pd.read_csv(path)

# Preprocesa texto eliminando caracteres no deseados y aplicando lematización
def preprocess_tweet(tweet, stop_words, lemmatizer):
    """
    Preprocesa un texto eliminando URLs, menciones, hashtags, caracteres no alfabéticos,
    stopwords y aplica lematización.

    Args:
        tweet (str): Texto a procesar.
        stop_words (set): Conjunto de palabras a eliminar.
        lemmatizer (WordNetLemmatizer): Objeto para lematización de palabras.

    Returns:
        str: Texto procesado.
    """
    if pd.isnull(tweet):
        return ''
    tweet = re.sub(r"http\S+|www\S+|https\S+|\@\w+|\#|[^A-Za-z\s]", '', tweet)  # Limpieza
    tweet = tweet.strip().lower()  # Normalización
    tokens = word_tokenize(tweet)  # Tokenización
    tokens = [word for word in tokens if word not in stop_words]  # Eliminación de stopwords
    if not tokens:
        return ''
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lematización
    return ' '.join(tokens)

# Balancea un dataset entre clases
def balance_dataset(dataset, label_column, positive_label='positive', negative_label='negative', random_state=42):
    """
    Balancea un dataset igualando la cantidad de ejemplos entre clases.

    Args:
        dataset (pandas.DataFrame): Dataset original.
        label_column (str): Nombre de la columna con etiquetas.
        positive_label (str): Etiqueta de la clase positiva.
        negative_label (str): Etiqueta de la clase negativa.
        random_state (int): Semilla para garantizar reproducibilidad.

    Returns:
        pandas.DataFrame: Dataset balanceado.
    """
    positive_rows = dataset[dataset[label_column] == positive_label]
    negative_rows = dataset[dataset[label_column] == negative_label]
    num_negative = negative_rows.shape[0]
    positive_sample = positive_rows.sample(n=num_negative, random_state=random_state)
    balanced_dataset = pd.concat([positive_sample, negative_rows], ignore_index=True)
    return balanced_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

# Vectoriza texto utilizando TF-IDF
def vectorize_data(dataset, text_column='cleaned_text'):
    """
    Convierte texto en vectores numéricos usando TF-IDF.

    Args:
        dataset (pandas.DataFrame): Dataset que contiene el texto a vectorizar.
        text_column (str): Columna con el texto.

    Returns:
        tuple: Vectorizador TF-IDF entrenado y matriz TF-IDF transformada.
    """
    tfidf_vect = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X = tfidf_vect.fit_transform(dataset[text_column])
    return tfidf_vect, X

# Divide datos en entrenamiento y prueba
def split_dataset(X, dataset, label_column):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Args:
        X (sparse matrix): Representaciones numéricas de los datos.
        dataset (pandas.DataFrame): Dataset original.
        label_column (str): Columna con las etiquetas.

    Returns:
        tuple: Conjuntos de datos para entrenamiento y prueba (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, dataset[label_column], test_size=0.2, random_state=42)

# Realiza búsqueda de hiperparámetros y entrena el modelo SVM
def optimize_svm(train_X, test_X, train_Y, test_Y):
    """
    Realiza búsqueda de hiperparámetros para un modelo SVM y evalúa su desempeño.

    Args:
        train_X (sparse matrix): Datos de entrenamiento.
        test_X (sparse matrix): Datos de prueba.
        train_Y (array-like): Etiquetas de entrenamiento.
        test_Y (array-like): Etiquetas de prueba.

    Returns:
        tuple: Mejor modelo SVM entrenado, predicciones y precisión.
    """
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'loss': ['hinge', 'squared_hinge'],
        'penalty': ['l2'],
        'max_iter': [1000, 2000, 5000],
        'tol': [1e-4, 1e-3, 1e-2],
        'class_weight': [None, 'balanced', {'positive': 1, 'negative': 2}]
    }
    grid_search = GridSearchCV(LinearSVC(class_weight='balanced'), param_grid, cv=5, n_jobs=-1, verbose=3)
    grid_search.fit(train_X, train_Y)
    print(f"Mejores hiperparámetros: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    best_model.fit(train_X, train_Y)
    predictions = best_model.predict(test_X)
    accuracy = round(accuracy_score(test_Y, predictions) * 100, 1)
    print(f"Accuracy: {accuracy}%")
    print("Reporte de clasificación:\n", classification_report(test_Y, predictions))
    return best_model, predictions, accuracy

# Gráficos de resultados
def plot_confusion_matrix(test_Y, predictions):
    """
    Muestra la matriz de confusión de las predicciones.

    Args:
        test_Y (array-like): Etiquetas reales.
        predictions (array-like): Predicciones del modelo.
    """
    conf_matrix = confusion_matrix(test_Y, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=sns.light_palette("#aec7e8", as_cmap=True), cbar=False,
                xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'],
                linewidths=0.5, linecolor='black', annot_kws={"size": 10})
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('True', fontsize=10)
    plt.title('Confusion Matrix', fontsize=12, weight='bold')
    plt.show()

def plot_classification_report(test_Y, predictions):
    """
    Genera un gráfico con métricas de precisión, recall y F1-score.

    Args:
        test_Y (array-like): Etiquetas reales.
        predictions (array-like): Predicciones del modelo.
    """
    report = classification_report(test_Y, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    metrics = ['precision', 'recall', 'f1-score']
    report_df = report_df[metrics]
    report_df.plot(kind='bar')
    plt.show()

def plot_accuracy_bar(accuracy):
    """
    Muestra un gráfico de barra con la precisión del modelo.

    Args:
        accuracy (float): Precisión del modelo.
    """
    plt.bar(['Accuracy'], [accuracy])
    plt.show()

# Punto de entrada principal del script
if __name__ == "__main__":
    # Configuración inicial
    setup_nltk()
    stop_words = initialize_stopwords()
    lemmatizer = WordNetLemmatizer()

    # Carga y preprocesamiento del dataset
    path = r"/SVM_TFG/datasets/all_datasets_reviews.csv"
    dataset = load_dataset(path)
    dataset = dataset[dataset['Sentiment'].isin(['positive', 'negative'])]  # Filtrado de clases
    dataset['cleaned_text'] = dataset['Text'].apply(lambda x: preprocess_tweet(x, stop_words, lemmatizer))
    dataset = dataset[dataset['cleaned_text'].str.strip() != ""]  # Eliminación de textos vacíos
    balanced_dataset = balance_dataset(dataset, label_column='Sentiment')

    # Vectorización y división en conjuntos
    tfidf_vect, X = vectorize_data(balanced_dataset)
    train_X, test_X, train_Y, test_Y = split_dataset(X, balanced_dataset, label_column='Sentiment')

    # Entrenamiento del modelo y evaluación
    best_model, predictions, accuracy = optimize_svm(train_X, test_X, train_Y, test_Y)

    # Guardado del modelo y vectorizador
    joblib.dump(best_model, 'svm_model.pkl')
    joblib.dump(tfidf_vect, 'tfidf_vectorizer.pkl')
    print("Modelo y vectorizador guardados correctamente.")

    # Visualización de resultados
    plot_confusion_matrix(test_Y, predictions)
    plot_classification_report(test_Y, predictions)
    plot_accuracy_bar(accuracy)
