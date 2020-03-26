import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors

#TODO: finish it!
model = Word2Vec.load("data/models/model.bin")
vocab = list(model.wv.vocab.keys())
X = np.array([model[wd] for wd in vocab])

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)


G = nx.Graph()
for wd in vocab:
    G.add_node(wd)

threshold = 0.60
edges = []
for wd in vocab:
    similars = model.wv.most_similar(wd, topn=20)
    similars = [e[0] for e in similars if e[1] > threshold]
    for w in similars:
        t = tuple(sorted((wd, w)))
        edges.append(t)

for edge in set(edges):
    G.add_edge(edge[0], edge[1])
