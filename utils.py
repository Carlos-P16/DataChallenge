import re
import unicodedata
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.tokenize import word_tokenize

stopwords = [
    "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un",
    "por", "con", "no", "una", "su", "para", "es", "al", "lo", "como", "más",
    "pero", "sus", "le", "ya", "o", "fue", "este", "ha", "sí", "porque", "esta",
    "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay",
    "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les",
    "ni", "contra", "otros", "ese", "eso", "ante", "esos", "algunas", "algo",
    "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas", "nosotras",
    "vosotros", "vosotras", "os", "mío", "mía", "míos", "mías", "tuyo", "tuya",
    "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra",
    "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos",
    "esas", "estoy", "estás", "está", "estamos", "estáis", "están", "esté",
    "estés", "estemos", "estéis", "estén", "estar", "estuve", "estuviste",
    "estuvo", "estuvimos", "estuvisteis", "estuvieron", "estuviera", "estuvieras",
    "estuviéramos", "estuvierais", "estuvieran", "estuviese", "estuvieses",
    "estuviésemos", "estuvieseis", "estuviesen", "estando", "estado", "estada",
    "estados", "estadas", "estad", "he", "has", "ha", "hemos", "habéis", "han",
    "haya", "hayas", "hayamos", "hayáis", "hayan", "habré", "habrás", "habrá",
    "habremos", "habréis", "habrán", "habría", "habrías", "habríamos", "habríais",
    "habrían", "había", "habías", "habíamos", "habíais", "habían", "hube",
    "hubiste", "hubo", "hubimos", "hubisteis", "hubieron", "hubiera", "hubieras",
    "hubiéramos", "hubierais", "hubieran", "hubiese", "hubieses", "hubiésemos",
    "hubieseis", "hubiesen", "habiendo", "habido", "habida", "habidos", "habidas",
    "soy", "eres", "es", "somos", "sois", "son", "sea", "seas", "seamos", "seáis",
    "sean", "seré", "serás", "será", "seremos", "seréis", "serán", "sería", "serías",
    "seríamos", "seríais", "serían", "era", "eras", "éramos", "erais", "eran", "fui",
    "fuiste", "fue", "fuimos", "fuisteis", "fueron", "fuera", "fueras", "fuéramos",
    "fuerais", "fueran", "fuese", "fueses", "fuésemos", "fueseis", "fuesen", "sintiendo",
    "sentido", "sentida", "sentidos", "sentidas", "siente", "sentid", "tengo", "tienes",
    "tiene", "tenemos", "tenéis", "tienen", "tenga", "tengas", "tengamos", "tengáis",
    "tengan", "tendré", "tendrás", "tendrá", "tendremos", "tendréis", "tendrán",
    "tendría", "tendrías", "tendríamos", "tendríais", "tendrían", "tenía", "tenías",
    "teníamos", "teníais", "tenían", "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis",
    "tuvieron", "tuviera", "tuvieras", "tuviéramos", "tuvierais", "tuvieran", "tuviese",
    "tuvieses", "tuviésemos", "tuvieseis", "tuviesen", "teniendo", "tenido", "tenida",
    "tenidos", "tenidas", "tened", 'paciente','tras','cm','realizo','presenta','dos',
    'presencia','mm','presentaba'
]

def clean_text(texto):
    """
    Limpia y preprocesa el texto

    Parameters:
    texto: corpus a limpiar y preprocesar

    Returns:
    None
    """
    # Pasar a minúsculas
    texto = texto.lower()
    
    # Eliminar signos de puntuación y numeros
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    
    # Eliminar acentos
    texto = ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))
    
    # Eliminar stop words
    palabras = texto.split()
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stopwords]
    
    # Unir palabras nuevamente
    texto_limpio = ' '.join(palabras_filtradas)
    
    return texto_limpio


def clean_corpus(df, column_name):
    """
    Limpia y preprocesa el texto de todo el dataframe

    Parameters:
    df (dataframe)
    column_name: Columna de df en la que está el corpus

    Returns:
    None
    """
    # Copiar el DataFrame para no modificar el original
    df_clean = df.copy()
    
    # Limpiar el texto en la columna especificada
    df_clean[column_name] = df_clean[column_name].apply(lambda x: clean_text(x))
    
    return df_clean

def plot_bow_heatmap(df, column_name, n_words, n_grams=1):
    """
    Visualiza los n_words términos más frecuentes de un Bag of Words (BoW) en un mapa de calor,
    agrupando los documentos de 20 en 20 y mostrando el total de conteos de cada palabra en esa fila del BoW.

    Parameters:
    df (DataFrame): El DataFrame que contiene el corpus de texto.
    column_name (str): El nombre de la columna en el DataFrame que contiene el corpus.
    n_words (int): Número de términos a representar en el mapa.
    n_grams (int): Número de palabras en cada n-grama (1 para unigramas, 2 para bigramas, 3 para trigramas, etc.).

    Returns:
    None
    """
    # Obtener el corpus de la columna especificada
    corpus = df[column_name]
    
    # Tokenizar el corpus en palabras
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
    
    # Entrenar el modelo de n-gramas
    ngram_model = Phrases(tokenized_corpus, min_count=5, threshold=100)
    for _ in range(n_grams-1):
        ngram_model = Phrases(ngram_model[tokenized_corpus], min_count=5, threshold=100)
    
    # Convertir el modelo en Phraser para mayor eficiencia
    ngram_phraser = Phraser(ngram_model)
    
    # Aplicar el modelo a los documentos
    ngram_corpus = [' '.join(ngram_phraser[doc]) for doc in tokenized_corpus]
    
    # Crear un vectorizador CountVectorizer
    vectorizer = CountVectorizer(max_features=n_words)
    
    # Obtener la matriz de conteo de palabras
    bow_matrix = vectorizer.fit_transform(ngram_corpus)
    
    # Obtener los nombres de las palabras
    words = vectorizer.get_feature_names_out()
    
    # Agrupar los documentos de 20 en 20 y sumar los conteos de cada palabra
    grouped_bow_matrix = np.zeros((len(corpus) // 20, len(words)))
    for i in range(len(corpus) // 20):
        start_index = i * 20
        end_index = (i + 1) * 20
        grouped_bow_matrix[i] = bow_matrix[start_index:end_index].sum(axis=0)
    
    # Convertir la matriz a un DataFrame de pandas
    bow_df = pd.DataFrame(grouped_bow_matrix, columns=words)
    
    # Crear el mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(bow_df, cmap="YlGnBu")
    plt.title(f'Top {n_words} Most Common Words Heatmap ({n_grams}-grams), Grouped by 20 Documents')
    plt.xlabel('Words')
    plt.ylabel('Document Groups (20 Documents each)')
    plt.show()