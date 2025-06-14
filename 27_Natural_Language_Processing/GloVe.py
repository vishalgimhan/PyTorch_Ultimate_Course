# %% Packages
import pandas as pd
from plotnine import ggplot, aes, geom_text, labs
from sklearn.manifold import TSNE
import torch
import torchtext.vocab as vocab

# %%
# https://nlp.stanford.edu/projects/glove/
glove = vocab.GloVe(name='6B', dim=100)

# %%
glove_dim = 1000

# %% Number of words and embeddings
glove.vectors.shape

# %%
print(glove.stoi['woman'])
print(glove.itos[0])

# %% Get ana embedding
def get_embedding_vector(word):
    word_index = glove.stoi[word]
    emb = glove.vectors[word_index]
    return emb
get_embedding_vector('chess').shape 

# %% Find closest words from input word
def get_closest_words_from_word(word, max_n=5):
    word_emb = get_embedding_vector(word)
    distances = [(w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item()) for w in glove.itos]
    dist_sort_filt = sorted(distances, key=lambda x: x[1])[:max_n]
    return dist_sort_filt

get_closest_words_from_word('chess')

#%% Find closest words from embedding
def get_closest_words_from_embedding(embedding, max_n=5):
    distances = [(w, torch.dist(embedding, get_embedding_vector(w)).cpu().item()) for w in glove.itos]
    dist_sort_filt = sorted(distances, key=lambda x: x[1])[:max_n]
    return dist_sort_filt

# %% FFind word analogies
# eg. King is to Queen like Man is to Woman
def get_word_analogy(word1, word2, word3, max_n=5):
    # logic w1 = king, ...
    # w1 - w2 + w3 --> w4
    word1_emb = get_embedding_vector(word1)
    word2_emb = get_embedding_vector(word2)
    word3_emb = get_embedding_vector(word3)
    word4_emb = word1_emb - word2_emb + word3_emb
    analogy = get_closest_words_from_embedding(word4_emb)
    return analogy

get_word_analogy(word1='sister', word2='brother', word3='nephew')

# %% Word Clusters
def get_closest_words_from_word_only_word(word, max_n=5):
    word_emb = get_embedding_vector(word)
    distances = [(w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item()) for w in glove.itos]
    dist_sort_filt = sorted(distances, key=lambda x: x[1])[:max_n]
    return [item[0] for item in dist_sort_filt]

words = []
categories = ['numbers', 'algebra', 'music', 'science', 'technology']

df_word_cloud = pd.DataFrame({
    'category': [],
    'word': []
})

for category in categories:
    print(category)
    closest_word = get_closest_words_from_word_only_word(word=category, max_n=20)
    temp = pd.DataFrame({
        'category': [category] * len(closest_word),
        'word': closest_word
    })
    df_word_cloud = pd.concat([df_word_cloud, temp], ignore_index=True)
df_word_cloud

# %% Get 100 dimensiion word embeddings for all words
n_rows = df_word_cloud.shape[0]
n_cols = glove_dim
X = torch.empty((n_rows, n_cols))
for i in range(n_rows):
    current_word = df_word_cloud.loc[i, 'word']
    X[i, :] = get_embedding_vector(current_word)
    print(f"{i}: {current_word}")

# %%
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X.cpu().numpy())

# %%
df_word_cloud['x'] = X_tsne[:, 0]
df_word_cloud['y'] = X_tsne[:, 1]

ggplot(data=df_word_cloud.sample(25)) + \
    aes(x='x', y='y', label='word', color='category') + \
    geom_text() + \
    labs(title='GloVe Word Embeddings and Categories')

# %%