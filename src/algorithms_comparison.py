# Libraries
# ====================================================
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import numpy as np
from scipy import sparse
import pickle
import time

# Data Import
df = pickle.load(open('../data/input_files/input','rb'))
tweets = df.text

# Frequent itemsets algorithms
# ====================================================

timings = {'Apriori mlxtend':[],
           'Apriori efficient': [],
           'Fp-growth':[]}

# Sparse matrix generation
#!pip install mlxtend
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(tweets).transform(tweets,sparse=True)
apriori_input = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

# 1. Apriori algorithm of mlxtend (the one I used)
def compute_apriori_mlxtend():
    from mlxtend.frequent_patterns import apriori

    start_time = time.time()
    frequent_itemsets = apriori(apriori_input, min_support=0.01,use_colnames=True,low_memory=True)
    timings['Apriori mlxtend'].append(time.time() - start_time)
    frequent_itemsets['n_items'] = [len(x) for x in frequent_itemsets.itemsets]

    return frequent_itemsets

# 2. Efficient apriori
#pip install efficient-apriori
def compute_efficient_apriori():
    transactions = [tuple(x) for x in tweets]
    from efficient_apriori import apriori

    start_time = time.time()
    itemsets, rules = apriori(transactions,min_support=0.01,min_confidence=0.8)
    timings['Apriori efficient'].append(time.time() - start_time)
    item_found = convert_result_to_df(itemsets)

    return item_found

def convert_result_to_df(itemsets):
    length = []
    itemset = []
    occurrence = []
    for i in range(1,max(itemsets)):
        for key in itemsets[i]:
            length.append(i)
            itemset.append(key)
            occurrence.append(itemsets[i][tuple(key)]/len(transactions))
    data = [length,occurrence,itemset]
    res = pd.DataFrame(columns=['n_items','occurrences','itemset'])
    res.n_items = length
    res.occurrences = occurrence
    res.itemset = itemset
    return res

# 3. FP-growth
def compute_fp_growth():
    from mlxtend.frequent_patterns import fpgrowth
    start_time = time.time()
    fp_result = fpgrowth(apriori_input, min_support=0.01,use_colnames=True)
    timings['Fp-growth'].append(time.time() - start_time)
    return fp_result

for i in range(100):
    compute_apriori_mlxtend()
    compute_fp_growth()
    compute_efficient_apriori()

# Timings comparison on the overall dataset
time = pd.DataFrame.from_dict(timings)
time.mean()

# Comparison between results
apriori_1 = compute_apriori_mlxtend().sort_values(['occurrences'],ascending=False).reset_index(drop=True)
apriori_2 = compute_efficient_apriori().sort_values(['support','n_items'],ascending=[False,True]).reset_index(drop=True)
apriori_3 = compute_fp_growth().sort_values(['support'],ascending=False).reset_index(drop=True)

# All of them return the exact same itemsets with the same support
pd.concat([apriori_1,apriori_2,apriori_3],axis=1).drop(['n_items'],1)
