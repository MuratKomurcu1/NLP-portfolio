import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import re
from collections import Counter

from IPython.display import display
from gensim.models import Word2Vec


#Homework Dataset
data = {
    "text": [
        "Manchester United won the match 2-1 after a thrilling game.",
        "The new smartphone model includes AI-based camera features.",
        "The president addressed the nation regarding the new policy.",
        "Regular exercise helps reduce the risk of heart diseases.",
        "The movie received positive reviews from critics and audiences.",
        "Serena Williams announced her retirement from tennis.",
        "Apple is planning to launch its VR headset next year.",
        "Government officials discussed the climate change agreement.",
        "A balanced diet and sleep are essential for good health.",
        "The upcoming film features several Oscar-winning actors.",
        "The team celebrated their championship victory with fans.",
        "Researchers developed a new algorithm for speech recognition.",
        "Parliament passed a new law to support small businesses.",
        "Doctors recommend regular check-ups for early diagnosis.",
        "The actor signed a contract for three upcoming movies."
    ],
    "label": [
        "Sports",
        "Technology",
        "Politics",
        "Health",
        "Entertainment",
        "Sports",
        "Technology",
        "Politics",
        "Health",
        "Entertainment",
        "Sports",
        "Technology",
        "Politics",
        "Health",
        "Entertainment"
    ]
}

 # Create DataFrame
df = pd.DataFrame(data)

print(df.head())
print(df.tail())
print(df.shape) 

"""
                                                text          label
0  Manchester United won the match 2-1 after a th...         Sports
1  The new smartphone model includes AI-based cam...     Technology
2  The president addressed the nation regarding t...       Politics
3  Regular exercise helps reduce the risk of hear...         Health
4  The movie received positive reviews from criti...  Entertainment
                                                 text          label
10  The team celebrated their championship victory...         Sports
11  Researchers developed a new algorithm for spee...     Technology
12  Parliament passed a new law to support small b...       Politics
13  Doctors recommend regular check-ups for early ...         Health
14  The actor signed a contract for three upcoming...  Entertainment

(15, 2)
"""

print(df["label"].value_counts())
"""
        label
    Sports           3
    Technology       3
    Politics         3
    Health           3
    Entertainment    3
    Name: count, dtype: int64
    
    Ve her kategoriden 3 metin var. Bu da dengeli, güzel bir veri seti demek.
"""
    
# BOW Vectorization

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

# Convert to DataFrame for better visualization
bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Display the BOW DataFrame
display(bow_df)

print("-------------------------------------------------------------------   ")

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(df['text'])

# Convert to DataFrame for better visualization
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Display the TF-IDF DataFrame
display(tfidf_df)

print("-------------------------------------------------------------------   ")

# N-grams
n_gram_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
X_ngrams = n_gram_vectorizer.fit_transform(df['text'])

ngram_df = pd.DataFrame(X_ngrams.toarray(), columns=n_gram_vectorizer.get_feature_names_out())

# Display the N-grams DataFrame
display(ngram_df)

print("-------------------------------------------------------------------   ") 

# Word2Vec Embeddings

import re

def temizle(text):
    return re.sub(r"[^\w\s]", "", text.lower())  # Küçük harfe çevir, noktalama temizle

sentences = df['text'].apply(lambda x: temizle(x).split()).tolist()


sentences = df['text'].apply(lambda x: x.lower().split()).tolist()
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=1)

# Display the Word2Vec model's vocabulary
word2vec_vocab = pd.DataFrame(word2vec_model.wv.index_to_key, columns=['word'])
display(word2vec_vocab)
print(word2vec_model.wv.most_similar("movie"))