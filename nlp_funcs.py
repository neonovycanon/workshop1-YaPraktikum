import re
import numpy as np
import string
from scipy import spatial

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

def clean_text(text, tokenizer, stopwords):
    """Обработка текстовых данных, генерация токенов
    Аргументы:
        text: текст, подвергающийся токенизации
    Выходные данные:
        Токенизированный текст в виде структуры python list
    Source: https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/#clean-and-tokenize-data
    """
    # Очистка текста для токенизации
    text = str(text).lower()  # приведение к одному регистру
    text = re.sub(r"\[(.*?)\]", "", text)  # удаление лишних символов [+XYZ chars]
    text = re.sub(r"\s+", " ", text)  # удаление нескольких пробелов
    text = re.sub(r"\w+…|…", "", text)  # удаление повторений
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # замена тире
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", "", text
    )  # удаление пунктуации

    tokens = tokenizer(text)  # токенизация
    tokens = [t for t in tokens if not t in stopwords]  # удаление стоп слов
    tokens = ["" if t.isdigit() else t for t in tokens]  # удаление цифр
    tokens = list(set([t for t in tokens if len(t) > 1]))  # удаление токенов длиной меньше 1
    return tokens

def vectorize(element, model):
    """Векторизация элемента с помощью эмбеддингов
    Аргументы:
        element: элемент для векторизации
        model: загруженная модель со встроенными эмбеддинагми
    Возвращаемые данные:
        Векторное представление элемента в виде эмбеддингов большой модели
    """
    zero_vector = np.zeros(model.vector_size) #создаём нулевой вектор
    vectors = [] #подготавливаем массив для хранения всех векторизированных токенов 
    for token in element: #цикл получения эмбеддингов для токенов текста 
        if token in model:
            try:
                vectors.append(model[token])
            except KeyError:
                continue
    if vectors: #возврат извлчённых признаков после использования эмбеддингов 
        vectors = np.asarray(vectors)
        avg_vec = vectors.mean(axis=0)
        return avg_vec
    else:
        return zero_vector #либо возврат нулевого вектора в случае отсутствия достаточных данных об эмбеддингах

def semantic_similarity(element, sim_dict):
  '''Вычисление семантической близости текстовых элементов
  Аргументы:
      element: элемент, содержащий эмбеддиниги текста
      sim_dict: словарь категорий, предназначенный
  Возвращаемые данные:
      1. Наиболее подходящая категория по самому максимальному значению косинусного расстояния;
      2. Либо общая категория 'Generic' в случае, если данных недостаточно для приведения к одной категории'''
  dist = []
  if np.any(element):
      for i in sim_dict.keys():
          emb = sim_dict.get(i)['embedding'][0]
          dist.append([i, 1-spatial.distance.cosine(element, emb)])
      dist = np.array(dist)
      similar_id = np.argmax(dist[:, 1])
      return dist[similar_id]
  else:
      return ['Generic', 0]