import pandas as pd
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle

# Find the most common word in the cleaned dataset
df = pickle.load(open('../data/input','rb'))
temp = []
for l in df.text:
    for el in l:
        temp.append(el)
temp
dic = dict(Counter(temp))
dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
word_count = pd.DataFrame(dic,columns=['term','count'])

fig = px.bar(word_count[:20],x='term',y='count', template='simple_white')
fig.update_layout(
    title={
        'text':"<b>The 20 most occurring terms inside the dataset</b>",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
         },
    xaxis={
        'title':"<b>Terms</b>",
    },
    yaxis={
        'title':"<b>Occurrences in the dataset</b>",
    },
    #plot_bgcolor='#191414',
    font=dict(
        family='Montserrat,Sans Serif',
        size=13,
        #color="#"
    )
)

df_day = df.groupby('date').sum()
df_day['item'] = [len(x) for x in df_day.text]
df_day = df_day.reset_index()
df_day.sort_values('item',ascending=False)

# Days in which we can find the highest number of tweets
df = pd.read_csv('../data/covid19_tweets.csv')
import datetime
df['date'] = pd.to_datetime(df.created_at,format='%Y-%m-%d').dt.date
df.groupby('date').count()['tweet']


# Clustering
# ========================================
with open('../data/input','rb') as f:
    df = pickle.load(f)
tweet = [' '.join(x) for x in df.text]

# Creating tfidf matrix
vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0)
tfidf = vectorizer.fit_transform(tweet)

# Elbow Method
#   Iterate to find the optimal number of clusters
values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=0)
    kmeans.fit(tfidf)
    values.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1,11), values)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Model fitting
model = KMeans(n_clusters=6, init='k-means++', max_iter=100, n_init=1)
model.fit(tfidf)

# Visualizing top terms per cluster
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(6):
    print("Cluster %d:" % i)
    temp = []
    for ind in order_centroids[i, :10]:
        temp.append(terms[ind])
    print(temp)

# GRAPH
#   Graph visualization of frequent itemsets
# ==========================================
import pickle
with open('../data/output', 'rb') as f:
    df = pickle.load(f)

edges = list([tuple(x) for x in df.itemsets])
#pip install networkx
import networkx as nx
G = nx.Graph()
# lone terms are self loops
edges_1 = [tuple(list(x)*2) for x in edges if len(x) ==1]
G.add_edges_from(edges_1)
# couples are ok
edges_2 = [x for x in edges if len(x) == 2]
G.add_edges_from(edges_2)
# triples need to be divided into 3: AB, BC, AC
import itertools
edges_3 = [x for x in edges if len(x) == 3]
couples = [list(itertools.combinations(x, 2)) for x in edges_3]
flat_list = [item for sublist in couples for item in sublist]
G.add_edges_from(flat_list)

# Visualization :)
def generate_adjlist_with_all_edges(G, delimiter=' '):
    temp = dict()
    for s, nbrs in G.adjacency():
        neigh =  [sublist for sublist in list(nbrs)]
        temp[s] = neigh
    return temp

adj = generate_adjlist_with_all_edges(G)
adj
edges = G.edges
#pip install pyvis
from pyvis.network import Network
net = Network(height="100%", width="100%",
              bgcolor="#191414", font_color="white",
              notebook=True,
              directed = False)

def get_adjacency(l,key):
    res = "<b>"+str(key)+"</b>:<br> "
    for item in l[key]:
        res+=item+"<br> "
    return res,len(l[key])

net.barnes_hut()
for couple in edges:
    src = couple[0]
    des = couple[1]
    title_src, n_src= get_adjacency(adj,src)
    title_des, n_des= get_adjacency(adj,des)
    if src == des and n_src<=1:
        color_src = '#09BC8A'
        color_des = color_src
    else:
        if n_src > 9:
            color_src = '#CC5A71'
        else:
            color_src = '#DAB785'
        if n_des > 9:
            color_des = '#CC5A71'
        else:
            color_des = '#DAB785'
    net.add_node(des, size = 70, color = color_des, title = title_des)
    net.add_node(src, size = 70, color = color_src, title = title_src)
    net.add_edge(src, des, width =20, arrowStrikethrough = False)


net.show('../data/frequent_itemsets_COVID19.html')
