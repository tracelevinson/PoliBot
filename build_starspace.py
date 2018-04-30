"""
build_starspace.py

Save finalized article training/test datasets and build Starspace embeddings.

Input: articles_full_detailed.tsv (clean articles dataset from scrape_url_text.py)
Outputs:
1) articles_ss_train_detailed.tsv, articles_ss_test_detailed.tsv (final detailed article training/test sets)
2) ss_embeddings.tsv (individual word embeddings; direct Starspace model output)
3) article_embeddings.pkl (full article embeddings for each article in dataset)
"""
###
from utils import *
###
full_detailed_path = home + 'data/articles_full_detailed.tsv'
full_detailed_pandas_path = home + 'data/articles_full_detailed_pandas.csv'
ss_embeddings_path = home + 'results/ss_embeddings.tsv'
###

# Pull in raw dataset for cleaning
df = pd.read_csv(full_detailed_path, encoding = 'ISO-8859-1', sep='\t', header=None)

# Split articles dataset in to train and test sets (80/20). Limit to concat feature for exportation.
df['testinclude'] = np.random.randn(len(df))
df['testinclude'] = df.testinclude > 0.8
df_train = df[df.testinclude==0].drop('testinclude', axis=1)
df_test = df[df.testinclude==1].drop('testinclude', axis=1)

# Export datasets to TSV files as inputs to StarSpace embeddings
df_train.to_csv(home + 'data/articles_ss_train_detailed.tsv', sep='\t', header=False, index=False)
df_test.to_csv(home + 'data/articles_ss_test_detailed.tsv', sep='\t', header=False, index=False)

"""
Note: Separate TSV files are subsequently created within Excel to include only text and format to multiple columns, dropping '_detailed' file suffix on final train/test/full data.
"""

# Save additional basedoc file for Starspace ranking assessment (same as testFile here)
df_test.to_csv(home + 'data/test_base_docs.tsv', sep='\t', header=False, index=False)

"""
Create document-level StarSpace embeddings...
Process description: trainMode 2 enables an unsupervised learning StarSpace model, where one sentence per article is randomly selected as input, remaining sentences in that article become the positive label, and a random subset of other articles' text in the corpus provide negative examples (# specified). Each article acquires a document-level embedding with which to compare against incoming user query embeddings.
"""

# Call bash script to train, test, and evaluate Starspace embeddings
subprocess.call("./starspace.sh")

# Create and store full set of article embeddings to pickle file.
df = pd.read_csv(full_detailed_pandas_path, encoding = 'ISO-8859-1', header=0)
embeddings, dim_embed = load_embeddings(home + 'results/ss_embeddings.tsv')
df['embedding'] = df.text.apply(lambda x: vectorize_text(x, embeddings, dim_embed))
embed_array = df[['url','source','embedding']].as_matrix().transpose()
embed_array.dump(home + 'results/article_embeddings.pkl')
