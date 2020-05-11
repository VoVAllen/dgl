from data import MovieLens
ds = MovieLens('ml-100k', 'cpu', use_one_hot_fea=True)
g = ds.train_dec_graph
from sample import sample_graph

gg = sample_graph(g, 1)
print(gg)

