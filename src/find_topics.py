from text_processing import clean_tweets,import_dataset
from apriori_complete import find_topics,compute_apriori,filter_frequent_itemsets
from ast import literal_eval
import pandas as pd
import os
import time
import pickle # for saving and importing results

directory = 'USA_election_2020/'
dataset_name = 'hashtag_donaldtrump.csv'
df = pd.read_csv(directory+dataset_name,engine='python',encoding='utf-8')
df[['created_at','tweet']]
import_dataset(directory+dataset_name)

# find_popular_topics recaps all steps of the project,
# starting from text processing and returning apriori output
def find_popular_topics(dataset,support=0.1,name='output'):
    start_time = time.time()
    # Data import and cleaning
    if not os.path.isfile(directory+'clean_dataset.csv'):
        df = clean_tweets(import_dataset(dataset))
        print('File imported and cleaned')
    else:
        df = pd.read_csv(directory+'clean_dataset.csv')
        df.text = [literal_eval(x) for x in df.text]
        print('File imported')
    df.date = pd.to_datetime(df.date,format='%Y-%m-%d')

    result = find_topics(df,support) # Find topics day per day

    pickle.dump(result,open(name,'wb'))
    #result.to_csv('output.csv',index=False)
    print('Ended in '+str(time.time()-start_time)+'s')
    return result


output = find_popular_topics(directory+dataset_name,0.05,name=directory+'trump')

output = pd.read_csv('output.csv')

import pickle
results = pickle.load(open('output','rb'))

most_frequent = filter_frequent_itemsets(results,days=2,min_items=2,support=0.05).itemsets

for set in results.itemsets:
    if 'vaccin' in set:
        print(set)


output.itemsets[0]
