#-*- coding: utf-8 -*-

# Imports
import string
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

noms = ['../datasets/reviews_always.csv', '../datasets/reviews_gillette.csv',
        '../datasets/reviews_oral-b.csv', '../datasets/reviews_pantene.csv', '../datasets/reviews_tampax.csv']


def review_to_words(n):
    # Function to convert a raw review to a string of steemed words

    liste = pd.read_csv(noms[n], sep="\t", header=0) # We acquire the data
    data = []
    stops = set(stopwords.words("english")) # Stop words
    stemmer = SnowballStemmer("english") # Using Python Stemmer
    # for every line of our csv
    for index, row in liste.iterrows():
        review = str(row[7]) # We get the review
        rating = row[3] # The rating
        # 1. Remove non-letters        
        letters_only = re.sub("[^a-zA-Z]", " ", review) 
        #
        # 2. Convert to lower case, split into individual words
        words = letters_only.lower().split()

        # 3. Remove stop words and do stemming
        meaningful_words = [w for w in words if not w in stops]
        meaningful_words = [stemmer.stem(w) for w in meaningful_words]
        # 4. Join the words back into one string separated by space
        review = " ".join( meaningful_words )
        data.append((rating, review)) # Put the result in a list of tuples
    return data

# Missing comments can be found in model-1-knn.py file
def get_model(n):
    data = review_to_words(n)
    df = pd.DataFrame.from_records(data, columns=['user_rating', 'review'])
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None)
    df.sample(frac=1)
    tailleTrain = int((0.7) * len(df))
    tailleTest = len(df) - tailleTrain

    training = df[0:tailleTrain]  # Training
    testing = df[tailleTrain:]  # Testing

    train_data_features = vectorizer.fit_transform(training['review'])
    train_data_features = train_data_features.toarray()
    np.asarray(train_data_features)

    # Our classifier using Random Forest classifier
    # We are using 200 trees
    forest = RandomForestClassifier(n_estimators = 200, n_jobs=4)
    forest = forest.fit(train_data_features, training["user_rating"].tolist())
    
    test_data_features = vectorizer.transform(testing['review'])
    test_data_features = test_data_features.toarray()
    np.asarray(test_data_features)
    result = forest.predict(test_data_features)
    l = testing['user_rating'].tolist()
    rate = 0
    for i in range(len(testing)):
        if l[i] == result[i]:
            rate += 1
    s = 'Rate : ' + repr(100*rate/len(testing))
    print(s)
    to_write = 'Random Forest' + ", " + "file : " + noms[n] + ", " + "rate : " + repr(100*rate/len(testing))
    f = open('../Results/base_model_rate.txt', 'a')
    f.write(to_write)
    f.write("\n")
    f.close()
    output = pd.DataFrame( data={'user_rating':l, 'predicted':result} )
    s = '../Results/rf_base_model_' + repr(n) + '.csv'
    output.to_csv(s, sep="\t", index=False)

i = input('Enter a number (0-4) : ')
get_model(int(i))
