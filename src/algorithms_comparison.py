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

df_new = pd.read_csv('secure_file/03_processed_dataset.csv')
# Reading lists not as strings
from ast import literal_eval
df_new.text = [literal_eval(x) for x in df_new.text]

prova = df_new.text

# Frequent itemsets algorithms
# ====================================================

# Sparse matrix generation
#!pip install mlxtend
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(prova).transform(prova,sparse=True)

fi = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

timings = {}
# 1. Apriori algorithm
from mlxtend.frequent_patterns import apriori
import time

start_time = time.time()
frequent_itemsets = apriori(fi, min_support=0.01,use_colnames=True), #low_memory=True)
timings['Apriori_no_low'] = time.time() - start_time
timings
frequent_itemsets
#frequent_itemsets
#frequent_itemsets.to_csv('frequent_itemsets_005.csv',index=False)

#frequent_itemsets = pd.read_csv('frequent_itemsets.csv')
frequent_itemsets['n_items'] = [len(x) for x in frequent_itemsets.itemsets]

# 2. Efficient apriori
#pip install efficient-apriori
transactions = [tuple(x) for x in prova]
from efficient_apriori import apriori

start_time = time.time()
itemsets, rules = apriori(transactions,min_support=0.012,min_confidence=0.8)
timings['Efficient Apriori_15'] = time.time() - start_time
timings

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

item_found = convert_result_to_df(itemsets)

# 3. FP-growth
from mlxtend.frequent_patterns import fpgrowth
start_time = time.time()
fp_result = fpgrowth(fi, min_support=0.05,use_colnames=True)
fpgrowth(fi, min_support=0.01,use_colnames=True)
timings['FP-Growth_1'] = time.time() - start_time

# Timings comparison on the overall dataset
timings

# Comparison between results
apriori_1 = item_found.sort_values(['occurrences'],ascending=False).reset_index(drop=True)
apriori_2 = frequent_itemsets.sort_values(['support','n_items'],ascending=[False,True]).reset_index(drop=True)
apriori_3 = fp_result.sort_values(['support'],ascending=False).reset_index(drop=True)

# All of them return the exact same itemsets with the same support
pd.concat([apriori_1,apriori_2,apriori_3],axis=1).drop(['n_items'],1)
