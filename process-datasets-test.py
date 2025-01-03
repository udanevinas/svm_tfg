import kagglehub
import pandas as pd

# Función para cargar un dataset desde un archivo CSV
def load_dataset(path):
    """
    Carga un dataset desde un archivo CSV.

    Args:
        path (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame cargado con los datos del archivo.
    """
    dataset = pd.read_csv(path)
    return dataset

# Función para categorizar un puntaje en una categoría de sentimiento
def categorize_sentiment(target):
    """
    Clasifica un puntaje numérico en una categoría de sentimiento.

    Args:
        target (float): Puntaje numérico a categorizar.

    Returns:
        str: Categoría de sentimiento ('negative', 'neutral', 'positive', 'unknown').
    """
    if 1 <= target <= 2:
        return 'negative'
    elif 2 < target < 4:
        return 'neutral'
    elif 4 <= target <= 5:
        return 'positive'
    return 'unknown'

# Bloque principal del script
if __name__ == "__main__":
    # Definición de la ruta base para los datasets
    PATH_DIR = '/SVM_TFG/datasets/datasets/apple/'

    # Rutas específicas de los datasets
    path_iphone_se = PATH_DIR + 'iPhone_SE.csv'
    path_iphone_14 = PATH_DIR + 'iPhone_14.csv'
    path_iphone_16 = PATH_DIR + 'iPhone_16.csv'

    # Lista de rutas a los datasets
    PATHS_DATASETS = [
        path_iphone_se,
        # path_iphone_11,
        path_iphone_14,
        path_iphone_16
    ]

    # Diccionario con información de los datasets
    DATASETS = {
        'iPhone SE': {'Target': 'Ratings', 'Text': 'Reviews'},
        # 'iPhone 11': {'Target': 'Rating', 'Text': 'Review'},
        'iPhone 14': {'Target': 'Rating', 'Text': 'Review'},
        'iPhone 16': {'Target': 'Rating', 'Text': 'Review'}
    }

    # Cargar cada dataset y asignarlo a la clave correspondiente en DATASETS
    for (key, path) in zip(DATASETS.keys(), PATHS_DATASETS):
        DATASETS[key]['db'] = load_dataset(path)  # Carga del dataset

    # Confirmación de carga de datasets
    for database, data in DATASETS.items():
        print(f"Dataset cargado para {database}")

    # Renombrar columnas en cada dataset según las claves 'Target' y 'Text'
    for key, dataset_info in DATASETS.items():
        target_col = dataset_info['Target']
        text_col = dataset_info['Text']
        db = dataset_info['db']

        # Renombrar columnas para uniformidad
        db.rename(columns={target_col: 'Target', text_col: 'Text'}, inplace=True)

    # Combinar todos los DataFrames en uno solo
    combined_dataset = pd.DataFrame()  # DataFrame vacío para almacenar todos los datos
    for key, dataset_info in DATASETS.items():
        db = dataset_info['db']
        db['db'] = key  # Añadir columna indicando el origen del dataset
        combined_dataset = pd.concat([combined_dataset, db[['Target', 'Text', 'db']]], ignore_index=True)

    # Crear nueva columna "Sentiment" basada en la columna "Target"
    combined_dataset['Sentiment'] = combined_dataset['Target'].apply(categorize_sentiment)

    # Contar y mostrar la cantidad de cada categoría de sentimiento
    sentiment_counts = combined_dataset['Sentiment'].value_counts()
    print("\nCantidad de cada sentimiento:")
    print(sentiment_counts)

    # Guardar el dataset combinado en un archivo CSV
    combined_dataset.to_csv(PATH_DIR + 'iphone_reviews_processed.csv', index=False)
    print(f"El dataset procesado se ha guardado en {PATH_DIR}iphone_reviews_processed.csv")
