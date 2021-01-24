import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import numpy as np
from ast import literal_eval
import os
import time
import pickle # for saving and importing results
import sys

directory = '../data/testing/'
dataset_name = 'input'
output_dir = '../data/testing/'
try:
    n_items = int(sys.argv[0])
except:
    n_items=0


# Import of the dataset named 'input' in pickle format
def import_data():
    df = pickle.load(open(directory+dataset_name,'rb'))
    return df
import_data()

# Apriori algorithm on the given dataset with a specified support
def compute_apriori(df,support=0.05):
    # Timing
    start_time = time.time()

    # Encoder in a sparse matrix
    te = TransactionEncoder()
    te_ary = te.fit(df).transform(df,sparse=True)
    fi = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

    # Apriori algorithm
    frequent_itemsets = apriori(fi, min_support=support,use_colnames=True,max_len=3)
    print('\tExecution time:'+str(time.time()-start_time))
    frequent_itemsets['n_items'] = [len(x) for x in frequent_itemsets.itemsets]

    return frequent_itemsets

# Apriori algorithm for each day contained in the dataset, given the support
def compute_apriori_per_each_day(df, support=0.05):
    dates = df.date.unique()
    dates.sort()
    res = pd.DataFrame(columns = ['support','itemsets','n_items','date'])
    start = time.time()
    for date in dates:
        print('Day: '+str(date))
        temp=compute_apriori(df[df['date']==date].text,support)
        temp['date'] = [date for x in range(len(temp))]
        res = res.append(temp, ignore_index=True)
    print('Apriori Done in'+str(time.time()-start)+'sec.')
    return res

# Filtering the resulting dataframe of the apriori algorithm based on:
    # minimum support (default value=0.01)
    # minimum number of items (default value = 2)
    # minimum number of days (default value=1)
    # maximum number of items (default value = 4)
    # sorted per n_items and support
def filter_frequent_itemsets(df,support=0.01, min_items=2,max_items=4,days=1):
    filtered = df[(df['n_items']>=min_items) &
                  (df['support']>=support) &
                  (df['n_items']<=max_items) &
                  (df.date.apply(lambda x: len(x) > days))].reset_index(drop=True)
    return filtered.sort_values(['support','n_items'],ascending=[False,True]).reset_index(drop=True)

# Complete algorithm that computes apriori for each day
def find_topics(df, n_items=0):
    apriori_day = compute_apriori_per_each_day(df)
    print(apriori_day[:10])
    print('Saving the dataset...')
    #apriori_day.date = [x.strftime('%Y-%m-%d') for x in apriori_day.date]
    final = apriori_day.groupby(['itemsets','n_items']).agg({'date': lambda x: ','.join(x),'support':np.mean}).reset_index()
    final.date = [x.split(',') for x in final.date]
    final.itemsets = [list(x) for x in apriori_day.itemsets]
    pickle.dump(final,open(output_dir+'output','wb'))
    final.to_csv(output_dir+'output.csv',index=False)
    print('File saved.')

    # Showing only the requested number of lines
    if n_items==0 or n_items >len(final):
        n_items=len(final)
    print(final[:n_items])
    return final[:n_items]

# Call to the function that computes frequent itemsets
find_topics(import_data(),n_items)
