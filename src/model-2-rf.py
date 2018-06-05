#-*- coding: utf-8 -*-

# Imports
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
from sklearn.ensemble import RandomForestClassifier

verrou = RLock() # A lock for synchronised threads execution

noms = ['../datasets/reviews_always.csv', '../datasets/reviews_gillette.csv',
        '../datasets/reviews_oral-b.csv', '../datasets/reviews_pantene.csv',
        '../datasets/reviews_tampax.csv', '../datasets/reviews_short.csv']

# Here, we convert the word2vec data structure to a dataframe
f = open('../datasets/word2vec.txt', 'r+')
f.readline()
l0 = []
l1 = []
for line in f:
    ligne = line.split(" ")
    l0.append(ligne[0]) # The word
    l1.append(ligne[1:len(ligne) - 1]) # The vector of meaning
map(float, l1) # Changing the vector (string) to float
w2num = pd.DataFrame(data={'word': l0, 'meaning': l1}) # Dataframe based on word2vec
f.close()

inp = input('Enter a number (0-5) : ')
inp = int(inp)

# Here, we build our vector of semantic meaning
liste = pd.read_csv(noms[int(inp)], sep="\t", header=0)
data = []
stops = set(stopwords.words("english"))
mots = []
rates = []
places = []

# For every row of our database
for index, row in liste.iterrows():
    review = str(row[7]) # The review
    rating = row[3] # The rating
    # 1. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review)
    #
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    mots.append(meaningful_words) # A list of meaningful words
    places.append(index) # A list of review words to work on
    rates.append(rating) # A list of rating

# Thread used to build the vector of semantic meaning
class Threadify(Thread):
    """Thread for vector building"""

    def __init__(self, ind):
        Thread.__init__(self)
        self.ind = ind # Number of this thread

    def run(self):
        """Thread execution"""
        num = 0
        # While we have rows of review to work on
        while len(places) != 0:
            for i in range(len(places)): # We get the first available number to work on
                if (places[i] != -1):
                    with verrou: # With our lock
                        num = places[i] # We set the actual number to i
                        places.remove(num) # We remove i from the possible numbers
                    break
            
            words = mots[num] # We get the row corresponding to i
            init = np.zeros([1, 200], dtype=float) # We initialize a vector init(1, 200) of 0's
            taille = 0 # The size of the words having a meaning
            # For every word in the review
            for i in words:
                a = w2num.loc[w2num['word'] == i]['meaning'].tolist() # We get the vector of meaning
                # If it's in the dictionnary
                if len(a) > 0:
                    taille += 1
                    i = np.asarray(a, dtype=float)
                    init = np.add(init, i) # We add it to init
            a = np.zeros((1, 200), dtype=float)
            a = np.add(a, np.divide(init, taille)) # Finally, we get the mean of our vector of meaning
            with verrou:
                data.append((rates[num], a)) # We add the tuple (rating, meaning) to our data

# Here we decide to load written data from a file
lo = input('Load data ? (0/1) : ')
lo = int(lo)
if lo == 1:
    with open(noms[inp].replace(".csv", ".txt"), 'rb') as f:
        data = pickle.load(f) # We load our list of tuples
# If we don't want to load
else:
    threads = []
    # We create 4 threads for data acquiring
    for i in range(4):
        threads.append(Threadify(i))

    for i in range(4):
        threads[i].start() # We start the threads

    for i in range(4):
        threads[i].join() # We stop them and we write the data to a file
    with open(noms[inp].replace(".csv", ".txt"), 'wb') as f:
        pickle.dump(data, f)

def get_model(n):
    # This is our classifier using Random Forest method
    df = pd.DataFrame.from_records(data, columns=['user_rating', 'average_vec'])
    df.sample(frac=1) # We shuffle the data set
    tailleTrain = int((0.7) * len(df))
    tailleTest = len(df) - tailleTrain

    training = df[0:tailleTrain]  # Training set
    testing = df[tailleTrain:]  # Testing set

    # Our classifier with 200 estimators
    forest = RandomForestClassifier(n_estimators=200, n_jobs=4)
    x = np.reshape(training['average_vec'].tolist(), (len(training), 200)) # We get the list of mean
    forest = forest.fit(x, training["user_rating"].tolist()) # We fit the training to the ratings

    y = np.reshape(testing['average_vec'].tolist(), (len(testing), 200))
    result = forest.predict(y) # We get the prediction on test data
    l = testing['user_rating'].tolist()

    # We get the classification rate
    rate = 0
    for i in range(len(testing)):
        if l[i] == result[i]:
            rate += 1
    s = 'Rate : ' + repr(100 * rate / len(testing)) 
    print(s)

    # Writing results to files
    to_write = 'Random Forest' + ", " + "file : " + noms[inp] + ", " + "rate : " + repr(100*rate/len(testing))
    f = open('../Results/my_model_rate.txt', 'a')
    f.write(to_write)
    f.write("\n")
    f.close()
    output = pd.DataFrame(data={'user_rating': l, 'predicted': result})
    s = '../Results/rf_my_model_' + repr(inp) + '.csv'
    output.to_csv(s, sep="\t", index=False)

get_model(inp)