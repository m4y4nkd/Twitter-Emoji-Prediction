import re
import string
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
import winsound
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec


def strip_links(text):
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text


def strip_all_entities(text):
    entity_prefixes = ['@', '#']
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            text = text.replace(separator, ' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


tweet_list = []
labels = []
with open("us_training.text", "r", encoding="utf-8") as sentences_file:
    reader = sentences_file
    for row in reader:
        tweet_list.append(row)

with open("us_training.labels", "r") as sentences_file:
    reader = sentences_file
    for row in reader:
        labels.append(int(row))

for t in range(0, len(tweet_list)):
    tweet_list[t] = strip_all_entities(strip_links(tweet_list[t]))
    tweet_list[t] = tweet_list[t].replace("&amp;", "")
    tweet_list[t] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(^#)", " ", tweet_list[t]).split())

tzr = TweetTokenizer(preserve_case=False)
vocab = []
sentences = []
for t in tweet_list:
    sentences.append(tzr.tokenize(t))
    for token in tzr.tokenize(t):
        vocab.append(token)
vocab = list(set(vocab))

model = Word2Vec.load("test")
model.save("test")

sentence_vector = np.zeros((len(tweet_list), 100))
count_i = 0
for t in tweet_list:
    temp = np.zeros(100)
    for token in t.split():
        if token in model:
            temp = temp + model[token]
    for count_j in range(0, 100):
        sentence_vector[count_i][count_j] = temp[count_j]
    count_i += 1

X = sentence_vector
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


svc = SVC()
svc.fit(X_train, y_train)
file = open("english.output.txt", "w")
res = svc.predict(X)
for r in res:
    file.write(str(r)+"\n")
print(metrics.accuracy_score(y, res))


winsound.Beep(800, 400)
