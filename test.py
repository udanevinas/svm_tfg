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

# Configuración inicial y descargas necesarias para el procesamiento de texto con NLTK.
def setup_nltk():
    """
    Descarga los recursos necesarios de NLTK (stopwords, tokenizer y lematizador).

    Esta función no toma ni retorna parámetros.
    """
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

# Inicializa las stopwords y agrega términos adicionales.
def initialize_stopwords():
    """
    Inicializa la lista de stopwords en inglés y agrega palabras comunes de tweets.

    Returns:
        set: Conjunto de palabras stopwords.
    """
    stop_words = set(stopwords.words('english'))
    stop_words.update(['rt', 'via', 'us', 'u'])
    return stop_words

# Carga un dataset desde un archivo CSV.
def load_dataset(path):
    """
    Carga un dataset desde un archivo CSV.

    Args:
        path (str): Ruta del archivo CSV.

    Returns:
        pandas.DataFrame: Dataset cargado como un DataFrame.
    """
    return pd.read_csv(path)

# Preprocesa un tweet eliminando caracteres no deseados, stopwords y aplicando lematización.
def preprocess_tweet(tweet, stop_words, lemmatizer):
    """
    Preprocesa un texto eliminando URLs, menciones, hashtags, caracteres no alfabéticos,
    stopwords y aplicando lematización.

    Args:
        tweet (str): Texto del tweet a procesar.
        stop_words (set): Conjunto de stopwords a eliminar.
        lemmatizer (WordNetLemmatizer): Objeto lematizador para reducir las palabras a su forma base.

    Returns:
        str: Texto preprocesado.
    """
    if pd.isnull(tweet):
        return ''
    tweet = re.sub(r"http\S+|www\S+|https\S+|\@\w+|\#|[^A-Za-z\s]", '', tweet)  # Limpieza.
    tweet = tweet.strip().lower()  # Normalización.
    tokens = word_tokenize(tweet)  # Tokenización.
    tokens = [word for word in tokens if word not in stop_words]  # Eliminación de stopwords.
    if not tokens:
        return ''
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lematización.
    return ' '.join(tokens)

# Balancea un dataset igualando el número de clases.
def balance_dataset(dataset, label_column, positive_label='positive', negative_label='negative', random_state=42):
    """
    Balancea un dataset para igualar la cantidad de ejemplos positivos y negativos.

    Args:
        dataset (pandas.DataFrame): Dataset original.
        label_column (str): Nombre de la columna con las etiquetas.
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

# Convierte texto en vectores TF-IDF.
def vectorize_data(dataset, text_column='cleaned_text'):
    """
    Convierte texto en representaciones numéricas usando TF-IDF.

    Args:
        dataset (pandas.DataFrame): Dataset que contiene los textos.
        text_column (str): Nombre de la columna con los textos.

    Returns:
        tuple: Vectorizador TF-IDF entrenado y matriz TF-IDF transformada.
    """
    tfidf_vect = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X = tfidf_vect.fit_transform(dataset[text_column])
    return tfidf_vect, X

# Divide los datos en conjuntos de entrenamiento y prueba.
def split_dataset(X, dataset, label_column):
    """
    Divide los datos en entrenamiento y prueba.

    Args:
        X (sparse matrix): Representaciones numéricas de los datos.
        dataset (pandas.DataFrame): Dataset original.
        label_column (str): Nombre de la columna con las etiquetas.

    Returns:
        tuple: Conjuntos de entrenamiento y prueba (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, dataset[label_column], test_size=0.2, random_state=42)

# Entrena un modelo SVM optimizado.
def optimize_svm(train_X, test_X, train_Y, test_Y):
    """
    Entrena un modelo SVM con hiperparámetros predefinidos y evalúa su desempeño.

    Args:
        train_X (sparse matrix): Datos de entrenamiento.
        test_X (sparse matrix): Datos de prueba.
        train_Y (array-like): Etiquetas de entrenamiento.
        test_Y (array-like): Etiquetas de prueba.

    Returns:
        tuple: Modelo SVM entrenado, predicciones del modelo y precisión.
    """
    best_model = LinearSVC(
        C=100,
        loss='hinge',
        max_iter=5000,
        penalty='l2',
        tol=0.001,
        random_state=42,
        class_weight={'positive': 1, 'negative': 2}
    )
    best_model.fit(train_X, train_Y)
    predictions = best_model.predict(test_X)
    accuracy = round(accuracy_score(test_Y, predictions) * 100, 1)
    print(f"Accuracy: {accuracy}%")
    print("Reporte de clasificación:\n", classification_report(test_Y, predictions))
    return best_model, predictions, accuracy

# Genera y muestra una matriz de confusión.
def plot_confusion_matrix(test_Y, predictions):
    """
    Genera un gráfico de la matriz de confusión.

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

# Genera un gráfico con el reporte de clasificación.
def plot_classification_report(test_Y, predictions):
    """
    Genera un gráfico con las métricas de precisión, recall y F1-score.

    Args:
        test_Y (array-like): Etiquetas reales.
        predictions (array-like): Predicciones del modelo.
    """
    report = classification_report(test_Y, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    metrics = ['precision', 'recall', 'f1-score']
    report_df = report_df[metrics]
    plt.figure(figsize=(8, 5))
    ax = report_df.plot(kind='bar', color=['#fbb4b9', '#c2dff7', '#fcd8a5'], width=0.8)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=8, textcoords='offset points', xytext=(0, 5))
    plt.title('Classification Report', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1))
    plt.show()

# Genera un gráfico de barra mostrando la precisión del modelo.
def plot_accuracy_bar(accuracy):
    """
    Genera un gráfico de barra con la precisión del modelo.

    Args:
        accuracy (float): Precisión del modelo como porcentaje.
    """
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [accuracy], color='#c7e9f1', edgecolor='#a1c6ea')
    plt.text(0, accuracy + 2, f'{accuracy:.1f}%', ha='center', fontsize=8)
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.title('Model Accuracy', fontsize=12)
    plt.show()


# Punto de entrada principal del script.
if __name__ == "__main__":
    setup_nltk()  # Configura NLTK.
    stop_words = initialize_stopwords()  # Inicializa las stopwords.
    lemmatizer = WordNetLemmatizer()  # Crea un lematizador.

    # Cargar el modelo y el vectorizador previamente entrenados.
    best_model = joblib.load('svm_model.pkl')
    tfidf_vect = joblib.load('tfidf_vectorizer.pkl')
    print("Modelo y vectorizador cargados correctamente.")

    # Carga y preprocesamiento del dataset no visto.
    path = r"/SVM_TFG/datasets/apple/iphone_reviews_processed.csv"
    dataset_not_view = load_dataset(path)
    dataset_not_view = dataset_not_view[dataset_not_view['Sentiment'].isin(['positive', 'negative'])]
    dataset_not_view['cleaned_text'] = dataset_not_view['Text'].apply(lambda x: preprocess_tweet(x, stop_words, lemmatizer))
    dataset_not_view = dataset_not_view[dataset_not_view['cleaned_text'].str.strip() != ""]
    print("\nCantidad de cada sentimiento:\n", dataset_not_view['Sentiment'].value_counts())

    # Transforma el texto en vectores TF-IDF y evalúa el modelo.
    X_test_not_view = tfidf_vect.transform(dataset_not_view['cleaned_text'])
    Y_test_not_view = dataset_not_view['Sentiment']

    predictions_test_not_view = best_model.predict(X_test_not_view)
    accuracy_test_not_view = round(accuracy_score(Y_test_not_view, predictions_test_not_view) * 100, 1)
    print(f"Accuracy del test no visto: {accuracy_test_not_view}%")
    print("Reporte de clasificación del test no visto:\n", classification_report(Y_test_not_view, predictions_test_not_view))
    # Genera visualizaciones.
    plot_confusion_matrix(Y_test_not_view, predictions_test_not_view)
    plot_classification_report(Y_test_not_view, predictions_test_not_view)
    plot_accuracy_bar(accuracy_test_not_view)