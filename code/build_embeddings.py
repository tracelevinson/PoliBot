"""
build_embeddings.py

Builds article embeddings using final political article dataset.

Input: articles_final.tsv (clean articles dataset from scrape_url_text.py)
Outputs:
1) word_embeddings.bin (individual word embeddings for vectorizing user queries)
2) article_embeddings.pkl (full article embeddings for each article in dataset)
"""
###
from utils import *
from gensim.models import Doc2Vec, Phrases, KeyedVectors
from gensim.models.doc2vec import LabeledSentence
###
articles_path = home + 'data/articles_final.csv'
word_embed_path = home + 'results/word_embeddings.bin'
article_embed_path = home + 'results/article_embeddings.pkl'
###

# Pull in raw dataset for cleaning
df = pd.read_csv(articles_path)

# Create bigrams to feed into doc2vec model
bigram = Phrases(df.text.str.split().tolist())

# Prepare docs for model
article_docs = [LabeledSentence(bigram[text], url) for text, url in zip(df.text.str.split().tolist(), df.url.tolist())]

# Create, train, and save doc2vec model
model = Doc2Vec(dm=0, dbow_words=1, min_count=3, negative=5, hs=0, sample=1e-5, window=10, vector_size=100, workers=8)
model.build_vocab(article_docs)
model.train(article_docs, total_examples=model.corpus_count, epochs=10)
model.wv.save_word2vec_format(word_embed_path, binary=True)

# Create full set of article embeddings
embeddings_df = pd.DataFrame(index=range(len(article_docs)), columns=['url','source','embedding'])
for i in range(len(article_docs)):
    embeddings_df.url[i] = article_docs[i].tags
    embeddings_df.source[i] = df.source[i]
    embeddings_df.embedding[i] = model.infer_vector(article_docs[i].words)

# Store embeddings to pickle file
embeddings_array = embeddings_df.as_matrix().transpose()
embeddings_array.dump(article_embed_path)
