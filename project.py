# Libraries
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import numpy as np

# Data Import
df = pd.read_csv('covid19_tweets.csv',engine='python')
min(df['date'])
max(df['date'])
# Remove useless columns

places = df.user_location.unique()
ny = [x for x in places if str(x).__contains__('New York')]
ny
for place in places:
    if "NY" in place:
            print(place)

df = df.drop(['user_name','user_description','user_followers','user_friends','user_favourites','user_verified','user_location','user_created','source','is_retweet'],axis=1)
# Converts the first column from string to datetime format
df.date = pd.to_datetime(df.date)

df.to_csv('covid_updated.csv',index=False)


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = list(df.text)

def tokenize_and_stem(tweet):
    stemmer = SnowballStemmer('english')
    stopword = set(stopwords.words('english'))

    #Tokenizing
    frase = [word for sent in nltk.sent_tokenize(tweet) for word in nltk.word_tokenize(sent)]
    frase = [stemmer.stem(word) for word in frase]

    # Link removal
    link = re.compile(r'https.*')
    hashtag = re.compile(r'#.*')
    alphanum = re.compile(r'\W+')
    frase = [word for word in frase if not link.match(word)]
    temp=[]
    for word in frase:
        if hashtag.match(word):
            temp.append(word[1:])
        else:
            temp.append(word)
    #Remove stopwords
    frase = [word for word in frase if word not in stopword]
    prova = [word for word in frase if word]
    return frase,temp

for value in df.loc[:3,'hashtags']:
    print(value)

df
# Check if hashtags are contained inside text
for row in range(len(df)):
    if df.loc[row,'hashtags'] is not np.nan:
        if any(hashtag not in df.loc[row,'text'] for hashtag in df.loc[row,'hashtags']):
            print('hell '+str(row))



prova = [tokenize_and_stem(sentence) for sentence in corpus]
tokenize_and_stem(corpus[2])
tokenize_and_stem("Plays playful playing")
#!pip install mlxtend # Just in case you do not have it installed yet
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(prova).transform(prova,sparse=True)
fi = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(fi, min_support=0.001,use_colnames=True, low_memory=True)
frequent_itemsets.to_csv('frequent_itemsets.csv',index=False)

tokenize_and_stem(corpus[3])
couples = [list(x) for x in frequent_itemsets['itemsets']]
[x for x in couples if len(x)>1]
max(df.date)

frequent_itemsets['n_items'] = [len(x) for x in frequent_itemsets.itemsets]
frequent_itemsets[frequent_itemsets['n_items']>1].sort_values(['support','n_items'],ascending=[False,True])
