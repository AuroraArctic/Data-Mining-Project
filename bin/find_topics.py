import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import numpy as np
import os
import time
import pickle # for saving and importing results
import sys

# Define path names
directory = '../data/input_files/'
dataset_name = 'input'
output_dir = '../data/output_files/'

# Apriori algorithm on the given dataset with a specified support
def compute_apriori(df,support,max_items):
    # Timing
    start_time = time.time()

    # Encoder in a sparse matrix
    te = TransactionEncoder()
    te_ary = te.fit(df).transform(df)
    apriori_input = pd.DataFrame(te_ary, columns=te.columns_)

    # Apriori algorithm
    frequent_itemsets = apriori(apriori_input, min_support=support,use_colnames=True,max_len=max_items)
    #print('\tExecution time: '+str(time.time()-start_time))
    frequent_itemsets['n_items'] = [len(x) for x in frequent_itemsets.itemsets]

    return frequent_itemsets

# Apriori algorithm for each day contained in the dataset, given the support
def compute_apriori_per_each_day(df,support,max_items):
    dates = df.date.unique()
    dates.sort()
    res = pd.DataFrame(columns = ['support','itemsets','n_items','date'])
    start = time.time()
    for date in dates:
        print('> Day: '+str(date))
        temp=compute_apriori(df[df['date']==date].text,support,max_items)
        temp['date'] = [date for x in range(len(temp))]
        res = res.append(temp, ignore_index=True)

    print('Apriori Done in', "%.2f" % round(time.time()-start, 2), "seconds.")
    return res

# Filtering the resulting dataframe of the apriori algorithm based on:
    # minimum support (default value=0.01)
    # minimum number of items (default value = 1)
    # minimum number of days (default value=1)
    # maximum number of items (default value = 4)
    # sorted per n_items and support
def filter_frequent_itemsets(df,support,min_items,max_items=4,days=1):
    filtered = df[(df['n_items']>=min_items) &
                  (df['support']>=support) &
                  (df['n_items']<=max_items) &
                  (df.date.apply(lambda x: len(x) > days))].reset_index(drop=True)
    return filtered.sort_values(['support','n_items'],ascending=[False,True]).reset_index(drop=True)

# Complete algorithm that computes apriori for each day
def find_topics(df,support,lines,days,min_items,max_items):
    print('Starting Apriori...')
    apriori_day = compute_apriori_per_each_day(df,support,max_items)

    print('Saving the dataset...')
    final = apriori_day.groupby(['itemsets','n_items']).agg({'date': lambda x: list(x),'support': 'mean'}).reset_index()
    final.itemsets = [list(x) for x in final.itemsets]
    final = filter_frequent_itemsets(final,support, min_items, max_items, days)

    # Saving all lines
    pickle.dump(final,open(output_dir+'output','wb'))
    final.to_csv(output_dir+'output.csv',index=False)
    print('File saved.')

    # Printing only the requested number of n_items
    if lines!=0:
        if lines >len(final):
            lines=len(final)
        else:
            print(final[:lines])
    return final[:lines]


# 1. Import of the dataset named 'input' in pickle format
df = pickle.load(open(directory+dataset_name,'rb'))
print('Dataset imported.')

# 2. Parameters evaluation
parameter_dict = {}
argv = sys.argv[1:]
for user_input in argv:
    if "=" not in user_input:
        continue
    varname = user_input.split("=")[0]
    varvalue = user_input.split("=")[1]
    parameter_dict[varname] = varvalue

# Default values
support=0.02
lines=0
days=1
min_items = 1
max_items = 4

# Setting values inserted by the user
total_param = ['lines','support','days','min_items','max_items']
if 'lines' in parameter_dict.keys():
    lines = int(parameter_dict['lines'])
if 'support' in parameter_dict.keys():
    support = float(parameter_dict['support'])
if 'days' in parameter_dict.keys():
    days = int(parameter_dict['days'])
if 'min_items' in parameter_dict.keys():
    min_items = int(parameter_dict['min_items'])
if 'max_items' in parameter_dict.keys():
    max_items = int(parameter_dict['max_items'])

print('Parameters evaluated.')
# 3. Executing the algorithm
find_topics(df,support,lines,days,min_items,max_items)
