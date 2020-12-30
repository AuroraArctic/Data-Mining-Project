
from text_processing import clean_tweets,import_dataset
from apriori_complete import find_topics,compute_apriori
from ast import literal_eval
import pandas as pd
import os
import time

dataset_name = 'covid19_tweets.csv'

def find_popular_topics(dataset):
    start_time = time.time()
    # Data import and cleaning
    if not os.path.isfile('clean_dataset.csv'):
        df = clean_tweets(import_dataset(dataset))
        print('File imported and cleaned')
    else:
        df = pd.read_csv('clean_dataset.csv')
        df.text = [literal_eval(x) for x in df.text]
        print('File imported')
    df.date = pd.to_datetime(df.date,format='%Y-%m-%d')

    result = find_topics(df,0.1) # Find topics day per day
    result.to_csv('output.csv',index=False)
    print('Ended in '+str(time.time()-start_time)+'s')
    return result

find_popular_topics(dataset_name)
