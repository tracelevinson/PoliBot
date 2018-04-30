#!/bin/bash

############################################################################################################
# starspace.sh

# Bash script builds article embeddings using Starspace model framework.

# Inputs: articles_ss_train.tsv, articles_ss_test.tsv, test_base_docs.tsv (input data for training/testing)
# Output: ss_embeddings.tsv (word embeddings used to form final article embeddings)
############################################################################################################

train_path='.../PoliBot/data/articles_ss_train.tsv'
test_path='.../PoliBot/data/articles_ss_test.tsv'
base_docs_path='.../PoliBot/data/test_base_docs.tsv'
embed_path='.../PoliBot/results/ss_embeddings.tsv'

echo $train_path
# Train Starspace model
~/Starspace/starspace train -trainFile $train_path -model trainModel -trainMode 2 -adagrad 1 -ngrams 1 -epoch 10 -dim 100 -similarity cosine -minCount 2 -verbose 1 -negSearchLimit 50 -lr 0.01 -fileFormat 'labelDoc'

# Test Starspace model
~/Starspace/starspace test -testFile $test_path -model trainModel -basedoc $base_docs_path -trainMode 2 -adagrad 1 -ngrams 1 -epoch 10 -dim 100 -similarity cosine -minCount 2 -verbose 1 -negSearchLimit 50 -lr 0.01 -fileFormat 'labelDoc'

# Save word embeddings to final TSV file
mv trainModel.tsv $embed_path

# Further assess embeddings efficacy with query_predict utility function on specified sample queries (e.g. "How will Trump's administration respond to North Korean threats"); list top 5 ranked candidates
~/Starspace/query_predict $embed_path 5 $train_path
