import os
import glob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string                       # для удаления пунктуации
from num2words import num2words
import re                           #регулярки для цифр
from gensim.models import Word2Vec 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.cluster.elbow import kelbow_visualizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')

def replace_numb(text):
    def replace(match):
        number = int(match.group())  
        return num2words(number)    

    text_with_words = re.sub(r'\d+', replace, text)  
    return text_with_words

def preprocessing(text):
    
    parts = text.split("Text:",1)  # оставляю все что идет после text:
    if len(parts) > 1:
        text = parts[1]
        
    text = text.lower() # к нижнему регистру
    text = text.translate(str.maketrans('', '', string.punctuation)) # удаление пунктуации
    
    #не уверена ка лучше в предобработке поступить с числами, оставила вариант с заменой на слово
    #text = text.translate(str.maketrans('', '', '0123456789')) #удаление цифр
    text = replace_numb(text) # заменяем числа на слова
    
    tokens = word_tokenize(text) #токенизация
    tokens = [word for word in tokens if not word in stopwords.words('english')]
    
    lemmatizer = WordNetLemmatizer() # использую лемматизацию тк тексты маленькие и их не много
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens
 
def reading_text_and_prepr(path_to_folder):
    files = glob.glob(os.path.join(path_to_folder,"*.txt"))   
    prepr_texts = [] 
    for f in files:
        with open(f) as file:
            text = file.read()
            prepr_texts.append(preprocessing(text))          
    return prepr_texts


def cluster_selection(model, word_vectors, max_clusters=10):
    silhouette_scores = []
    inertia = []
    for n_clusters in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(word_vectors)   
        
        inertia.append(kmeans.inertia_)  # сохранение среднекв расст (инерции)
        silhouette_scores.append(silhouette_score(word_vectors, kmeans.labels_)) # посчитаем коэф силуэта
        
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters+1), inertia, marker='o')
    plt.title('метод локтя')
    plt.xlabel('кол-во кластеров')
    plt.ylabel('значение инерции')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters+1), silhouette_scores, marker='o')
    plt.title('Коэф силуэта')
    plt.xlabel('кол-во кластеров')
    plt.ylabel('коэф силуэта')
    
    plt.tight_layout()
    plt.show()
    
    word_vectors = np.array(word_vectors)
    kelbow_visualizer(KMeans(random_state=0), word_vectors, k=(2,11), timing=False)
    
'''
рассмотрим 2 метрики сразу: метод локтя и силуэта
согласо методу локтя следует выбирать то количество кластеров на котором сглаживается показатель инерции. это происходит
на 5-7 кластерах
по коэф силуэта нужно выбирать кол-во кластеров исходя из наибольшего значения коэффициента
в данном случае наибольший коэф силуэта соответствует 3 кластерам, однако далее идет значительный спад, следующий локальный максимум достигается на значении в 5 кластеров
поэтому в качестве компромисной оценки для 2х методов выбираю количество в 5 кластеров
 что подтверждается функцией kelbow_visualizer из модуля yellowbrick
 '''  

def main():
    folder = 'sampled_texts'
    texts = reading_text_and_prepr(folder)
    
    model = Word2Vec(texts, vector_size=100, window=5, min_count=2, workers=4, sg=0)
    model.train(texts, total_examples=model.corpus_count, epochs=15)
    

    avg_vectors = []
    for text in texts:
        word_vectors = [model.wv[word] for word in text if word in model.wv]
        if word_vectors:
            avg_vector = np.mean(word_vectors, axis=0)
            avg_vectors.append(avg_vector)
    
    cluster_selection(model, avg_vectors)
    
    num_clusters = 5  ### выше в комментариях обосновала почему
    
    print(f'Количество кластеров: {num_clusters}')
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans_model.fit(avg_vectors)
    

    new_text_path = 'someText.txt'
    with open(new_text_path) as f:
        cur_text = f.read()
        new_text = preprocessing(cur_text)
    
    new_text_vector = [model.wv[word] for word in new_text if word in model.wv]
    avg_new_text_vector = np.mean(new_text_vector, axis=0)
    
    cluster_new_text = kmeans_model.predict([avg_new_text_vector])
    print(f'Документ {new_text_path} принадлежит кластеру {cluster_new_text[0]}')

if __name__ == '__main__':
    main()