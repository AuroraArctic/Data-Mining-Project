import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from ast import literal_eval
from mlxtend.frequent_patterns import apriori
import time
import numpy as np
from alive_progress import alive_bar


df = pd.read_csv('clean_dataset.csv', encoding='utf-8')

df.text = [literal_eval(x) for x in df.text]
df.date = pd.to_datetime(df.date,format='%Y-%m-%d')
#df
def compute_apriori(df,support):
    # Timing
    start_time = time.time()

    # Encoder
    te = TransactionEncoder()
    te_ary = te.fit(df).transform(df,sparse=True)
    fi = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

    # Apriori algorithm
    frequent_itemsets = apriori(fi, min_support=support,use_colnames=True, low_memory=True)
    print('Execution time:'+str(time.time()-start_time))
    frequent_itemsets['n_items'] = [len(x) for x in frequent_itemsets.itemsets]

    return frequent_itemsets


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

def filter_frequent_itemsets(df,support=0.01, min_items=2,max_items=4,days=1):
    filtered = df[(df['n_items']>=min_items) & (df['support']>=support) & (df['n_items']<=max_items) & (df.date.apply(lambda x: len(x) > 2))].reset_index(drop=True)
    return filtered.sort_values(['support','n_items'],ascending=[False,True]).reset_index(drop=True)

def find_topics(df,support):
    apriori_day = compute_apriori_per_each_day(df,support)
    apriori_day.date = [x.strftime('%Y-%m-%d') for x in apriori_day.date]
    final = apriori_day.groupby(['itemsets','n_items']).agg({'date': lambda x: ','.join(x),'support':np.mean}).reset_index()
    final.date = [x.split(',') for x in final.date]
    final.to_csv('output.csv',index=False)
    print('File saved')
    return final

#compute_apriori_per_each_day(df.loc[:50],)
#find_topics(df[df['date']<'2020-07-28'],0.01)
# filter_frequent_itemsets(final,support=0.02,min_items=1,days=2)
