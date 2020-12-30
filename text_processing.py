# Libraries
# ====================================================
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import numpy as np
import datetime as dt

# Data Import
# ====================================================
df_name = 'covid19_tweets.csv'
def import_dataset(name):
    df = pd.read_csv(name,engine='python')
    df = df[['date','text']]
    df.date = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')
    return df

# Saving the new dataset
#df.to_csv('input.csv',index=False)
df = import_dataset(df_name)
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
    temp = [x.lower() for x in temp] # lower case
    temp = [re.sub(r"[^a-zA-Z']+", '', x) for x in temp] # removes symbols in words
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

def clean_tweets(df):
    temp = []
    for tweet in df.text:
        temp.append(text_processing(tweet))
    new_columns = {'date':df.date,'text':temp}
    df_new = pd.DataFrame(new_columns)
    df_new.to_csv('clean_dataset.csv',index=False)
    return df_new
