"""
intents_prep.py

Prepare intents dataset for classification model.

Inputs: (1) movie_dialogues.tsv (Cornell dialogue dataset compiled from movie captions)
        (2) articles_full.tsv (political query dataset comprised of sentences from political articles, also used for Starspace model)
Output: intents.tsv (dataset ready for TensorFlow intent classification model)
"""
###
from utils import *
###

# Specify sequence hyperparameters
MAX_LENGTH = 200
SAMPLE_SIZE = 200000

# Import dialogue data (pre-processed from another project)
dialogues = pd.read_csv(home + 'data/movie_dialogues.tsv', sep='\t', header=0)

# Limit observations to reasonable query length with valid values
dialogues = dialogues[dialogues.text.str.len()<MAX_LENGTH].sample(SAMPLE_SIZE, random_state=0)

# Import political data
politics = (pd.read_csv(home + 'data/articles_full.tsv', encoding = 'ISO-8859-1', sep='\t', header=None)
              .stack(dropna=True) # combine tab-separated columns into one
              .reset_index()
              .ix[:,2] # remove hierarchical stack indexes
              .to_frame(name='text')) # re-establish pd.DataFrame from Series

# Limit observations to same length as dialogues for balanced classes. Choose from subset of political obs with reasonable query length.
politics = (politics[politics.text.str.len()<MAX_LENGTH]
            .sample(SAMPLE_SIZE, random_state=0))
politics['intent'] = 'political'

# Combine two datasets for intent classification
intents = dialogues.append(politics, ignore_index=True)
intents['text'] = intents.text.apply(process_text, args=(True,))

# Remove erroneous obs (few enough to keep balanced classes intact) and store to TSV
intents = intents[~((intents.text.isnull()) | (intents.text.str.contains('#NAME?')))]
intents.to_csv(home + 'data/intents.tsv', sep='\t', index=False)
