from ast import literal_eval
import pandas as pd
import os
import time
import pickle # for saving and importing results

directory = '../data/'
dataset_name = 'input'
# Import of the dataset
pickle.load(open(dataset_name,'rb'))

import_dataset(directory+dataset_name)

# find_popular_topics recaps all steps of the project,
# starting from text processing and returning apriori output
def find_popular_topics(dataset,support=0.1,name='../data/output'):
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

    # ==================================================
    print('Saving results in pickle and csv format')
    # Saving output in pickle format
    pickle.dump(result,open(name,'wb'))
    # Saving output in readable format (csv)
    result.to_csv('output.csv',index=False)
    print('Ended in '+str(time.time()-start_time)+'s')
    return result

output = find_popular_topics(directory+dataset_name,0.05)

output = pd.read_csv('output.csv')

results = pickle.load(open('output','rb'))

most_frequent = filter_frequent_itemsets(results,days=2,min_items=2,support=0.05).itemsets

for set in results.itemsets:
    if 'vaccin' in set:
        print(set)


# =========================================================
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from ast import literal_eval
from mlxtend.frequent_patterns import apriori
import time
import numpy as np
from alive_progress import alive_bar
import pickle



# This function computes apriori algorithm on the given DataFrame
# with the specified support
def compute_apriori(df,support):
    # Timing
    start_time = time.time()

    # Encoder
    te = TransactionEncoder()
    te_ary = te.fit(df).transform(df,sparse=True)
    fi = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

    # Apriori algorithm
    frequent_itemsets = apriori(fi, min_support=support,use_colnames=True,max_len=3)
    print('Execution time:'+str(time.time()-start_time))
    frequent_itemsets['n_items'] = [len(x) for x in frequent_itemsets.itemsets]

    return frequent_itemsets

# compute_apriori_per_each_day computes the apriori algorithm
# for each day inside the dataset, with a given support
def compute_apriori_per_each_day(df, support):
    dates = df.date.unique()
    dates.sort()
    res = pd.DataFrame(columns = ['support','itemsets','n_items','date'])
    for date in dates:
        print('Day: '+str(date))
        temp=compute_apriori(df[df['date']==date].text,support)
        temp['date'] = [date for x in range(len(temp))]
        res = res.append(temp, ignore_index=True)
    print('Apriori Done.')
    return res

# Filtering the resulting dataframe of the apriori algorithm
# based on minimum support, number of items and of days and the maximum number of items
# sorted per n_items and support
def filter_frequent_itemsets(df,support=0.01, min_items=2,max_items=4,days=1):
    filtered = df[(df['n_items']>=min_items) & (df['support']>=support) & (df['n_items']<=max_items) & (df.date.apply(lambda x: len(x) > 2))].reset_index(drop=True)
    return filtered.sort_values(['support','n_items'],ascending=[False,True]).reset_index(drop=True)

# complete algorithm that computes apriori for each day
def find_topics(df,support):
    apriori_day = compute_apriori_per_each_day(df,support)
    apriori_day.date = [x.strftime('%Y-%m-%d') for x in apriori_day.date]
    final = apriori_day.groupby(['itemsets','n_items']).agg({'date': lambda x: ','.join(x),'support':np.mean}).reset_index()
    final.date = [x.split(',') for x in final.date]
    final.itemsets = [list(x) for x in apriori_day.itemsets]
    pickle.dump(final,open('../data/output','wb'))
    final.to_csv('../data/output.csv',index=False)
    print('File saved')
    return final

with open('../data/input','rb') as f:
    df = pickle.load(f)
df.date = pd.to_datetime(df.date,format='%Y-%m-%d')


compute_apriori_per_each_day(df,support=0.01)
find_topics(df,0.05)

with open('../data/output','rb') as f:
    output = pickle.load(f)
output.itemsets = [list(x) for x in output.itemsets]
output
list(output.itemsets[3])
filter_frequent_itemsets(output,min_items=1)[359:]
