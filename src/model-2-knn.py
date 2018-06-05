#-*- coding: utf-8 -*-

import string
import pandas as pd
import re
import nltk
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from threading import Thread, RLock
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Comments can be found in model-2-rf.py

verrou = RLock()

noms = ['../datasets/reviews_always.csv', '../datasets/reviews_gillette.csv',
        '../datasets/reviews_oral-b.csv', '../datasets/reviews_pantene.csv',
        '../datasets/reviews_tampax.csv', '../datasets/reviews_short.csv']

f = open('../datasets/word2vec.txt', 'r+')
f.readline()
l0 = []
l1 = []
for line in f:
    ligne = line.split(" ")
    l0.append(ligne[0])
    l1.append(ligne[1:len(ligne) - 1])
map(float, l1)
w2num = pd.DataFrame(data={'word': l0, 'meaning': l1})
f.close()

inp = input('Entrer un chiffre (0-4) : ')
inp = int(inp)

# Here, we build our vector of semantic meaning
liste = pd.read_csv(noms[inp], sep="\t", header=0)
data = []
stops = set(stopwords.words("english"))
mots = []
rates = []
places = []

for index, row in liste.iterrows():
    review = str(row[7])
    rating = row[3]
    # 1. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review)
    #
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    mots.append(meaningful_words)
    places.append(index)
    rates.append(rating)


class Threadify(Thread):
    """Thread chargé de construire un vecteur."""

    def __init__(self, ind):
        Thread.__init__(self)
        self.ind = ind

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        num = 0
        while len(places) != 0:
            for i in range(len(places)):
                if (places[i] != -1):
                    with verrou:
                        num = places[i]
                        places.remove(num)
                    break
            
            words = mots[num]
            init = np.zeros([1, 200], dtype=float)
            taille = 0
            for i in words:
                a = w2num.loc[w2num['word'] == i]['meaning'].tolist()
                if len(a) > 0:
                    taille += 1
                    i = np.asarray(a, dtype=float)
                    init = np.add(init, i)
            a = np.zeros((1, 200), dtype=float)
            a = np.add(a, np.divide(init, taille))
            with verrou:
                data.append((rates[num], a))
lo = input('Load data ? (0/1) : ')
lo = int(lo)
if lo == 1:
    with open(noms[inp].replace(".csv", ".txt"), 'rb') as f:
        data = pickle.load(f)
else:
    threads = []
    for i in range(4):
        threads.append(Threadify(i))

    for i in range(4):
        threads[i].start()

    for i in range(4):
        threads[i].join()
    with open(noms[inp].replace(".csv", ".txt"), 'wb') as f:
        pickle.dump(data, f)

def get_model(n):
    df = pd.DataFrame.from_records(data, columns=['user_rating', 'average_vec'])
    df.sample(frac=1)
    tailleTrain = int((0.7) * len(df))
    tailleTest = len(df) - tailleTrain

    training = df[0:tailleTrain]  # Training
    testing = df[tailleTrain:]  # Testing

    neigh = input('Enter the number of neighbours : ')
    neigh = int(neigh)

    # Our actual classifier using Knn, default metric is Minkwoski distance
    forest = KNeighborsClassifier(n_neighbors=neigh, n_jobs=4)
    x = np.reshape(training['average_vec'].tolist(), (len(training), 200))
    forest = forest.fit(x, training["user_rating"].tolist())

    y = np.reshape(testing['average_vec'].tolist(), (len(testing), 200))
    result = forest.predict(y) # Prediction
    l = testing['user_rating'].tolist()
    rate = 0
    for i in range(len(testing)):
        if l[i] == result[i]:
            rate += 1
    s = 'Rate : ' + repr(100 * rate / len(testing))
    print(s)
    to_write = 'Knn, neighbors : ' + repr(neigh) + ", " + "file : " + noms[inp] + ", " + "rate : " + repr(100*rate/len(testing))
    f = open('../Results/my_model_rate.txt', 'a')
    f.write(to_write)
    f.write("\n")
    f.close()
    output = pd.DataFrame(data={'user_rating': l, 'predicted': result})
    s = '../Results/knn_my_model_' + repr(inp) + '.csv'
    output.to_csv(s, sep="\t", index=False)

get_model(inp)