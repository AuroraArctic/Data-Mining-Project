# Libraries
# ====================================================
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import numpy as np

# Data Import
# ====================================================
df = pd.read_csv('covid19_tweets.csv',engine='python')

# Dropping all useless columns
df = df.drop(['user_name','user_description','user_followers','user_friends','user_favourites','user_verified','user_location','user_created','source','is_retweet','hashtags'],axis=1)

# Converts the first column from string to datetime format
df.date = pd.to_datetime(df.date)
dates = [x.strftime('%Y-%m-%d') for x in df.date]
df.date = dates

df

# Saving the new dataset
df.to_csv('input.csv',index=False)

# Tweet text pre-processing
# ====================================================
corpus = list(df.text)

# Remove internal and final links from tweets
def remove_links(text):
    temp = text.split(' ')
    temp = [re.sub('http.*','',x) for x in temp]
    return ' '.join(temp)

# Removes symbols, such as #,@,\n,\r,. and ...
def remove_symbols(text):
    replacement = (('#', ' '),('@', ' '),('\r', ' '),('\n', ' '),('\.',' '),('â€¦',''))
    for element in replacement:
        temp = re.sub(element[0], element[1], text)
        text = temp
    return text

# Converts text into list of strings, removes non-alphanumerical characters,
# converts words to lower case, removes extra symbols and empty strings
def only_text(text):
    temp = text.split(' ') # divides string into list of words
    alphanum = re.compile(r'\W+')
    temp = [x for x in temp if not alphanum.match(x)] # only alphanum
    temp = [x.lower() for x in temp if not link.match(x)] # lower case
    temp = [re.sub(r"[^0-9a-zA-Z-']+", '', x) for x in temp] # removes symbols in words
    temp = list(filter(None,temp)) # removes empty strings
    return temp

# Remove stopwords (too common words), after reducing them to the root
def remove_stopwords(text):
    stemmer = SnowballStemmer('english')
    stopword = set(stopwords.words('english'))
    new_sentence = [stemmer.stem(word) for word in text]
    new_sentence = [word for word in new_sentence if word not in stopword]
    return new_sentence

# Text Preprocessing
def text_processing(text):
    return remove_stopwords(only_text(remove_symbols(remove_links(text))))

temp = []
for tweet in df.text:
    temp.append(text_processing(tweet))

new_columns = {'date':df.date,'text':temp}
df_new = pd.DataFrame(new_columns)
#df_new.to_csv('03_processed_dataset.csv',index=False)

df_new = pd.read_csv('03_processed_dataset.csv')

prova = df_new.text

# Apriori algorithm
# ====================================================
#!pip install mlxtend # Just in case you do not have it installed yet
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(prova).transform(prova,sparse=True)
fi = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(fi, min_support=0.001,use_colnames=True, low_memory=True)
frequent_itemsets
frequent_itemsets.to_csv('frequent_itemsets.csv',index=False)

frequent_itemsets = pd.read_csv('frequent_itemsets.csv')

frequent_itemsets['n_items'] = [len(x) for x in frequent_itemsets.itemsets]
frequent_itemsets
frequent_itemsets[frequent_itemsets['n_items']>1].sort_values(['support','n_items'],ascending=[False,True])


# Associate texts with days.
df_per_day = df_new.groupby('date').agg(sum);
df_per_day = df_per_day.reset_index('date');


for pair in frequent_itemsets.itemsets:
    for i in range(len(df_per_day)):
        if set(pair).issubset(set(df_per_day['te']))
