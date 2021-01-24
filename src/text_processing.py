# Libraries
# ====================================================
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
# nltk.download('stopwords')
import re
import pickle
import numpy as np
import datetime as dt
import sys

# Data Import
# ====================================================
# Import of the dataset as csv
def import_dataset(name):
    df = pd.read_csv(name,engine='python')
    df = df[['created_at','tweet']]
    df = df.rename(columns={'created_at':'date','tweet':'text'})
    df.date = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')
    print('Dataset imported')
    return df

# Tweet text pre-processing
# ====================================================
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

# Text Preprocessing of the single tweet
def text_processing(text):
    return remove_stopwords(only_text(remove_symbols(remove_links(text))))

# Cleaning of the column text
def clean_tweets(df):
    temp = []
    print('Starting cleaning...')
    for tweet in df.text:
        temp.append(text_processing(tweet))
    new_columns = {'date':df.date,'text':temp}
    df_new = pd.DataFrame(new_columns)
    print('Dataset cleaned')
    return df_new

# Saving the new dataset
def save_results():
    df_name = str(sys.argv[1]) # Path of the dataset to process
    df = clean_tweets(import_dataset(df_name))
    print('Saving dataset...')
    df.to_csv('../data/input.csv',index=False) # Saving as csv to be readable
    with open('../data/input','wb') as f: # Saving in data type saving format
        pickle.dump(df,f)
    print('Done.')

# Erase comment in order to execute this script with command line
# save_results()
