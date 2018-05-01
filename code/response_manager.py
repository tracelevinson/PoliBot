"""
response_manager.py

Establishes responses to user queries based on two components:
1) Intent classification: classify general vs. political queries
2) Article ranking: in the case of political queries, determine and respond with most relevant articles for recommendation

Inputs: Article embeddings model (build_embeddings.py) and intent classification model (intent_classifier.py) output
Output: ResponseManager() class for use in run_bot.py
"""
###
import os
from sklearn.metrics.pairwise import pairwise_distances_argmin
from chatterbot import ChatBot
from gensim.models import KeyedVectors
from utils import *
from intent_classifier import *
###

class ArticleRanker(object):
    """ Generate article recommendations in response to user queries using pre-computed article embeddings """

    def __init__(self):
        print('Loading embeddings.')
        self.word_embeddings = KeyedVectors.load_word2vec_format(home+'results/word_embeddings.bin', binary=True)
        self.urls, self.sources, self.article_embeddings = unpickle(home + 'results/article_embeddings.pkl')
        self.dim_embed = len(self.article_embeddings[0])
        self.source_map = {i:source for i, source in enumerate(sorted(set(self.sources)),1)}
        print('Embeddings loaded.')

    # Rank articles to maximize cosine similarity between articles and user query. Returns K highest articles.
    def choose_article_recs(self, query, K=3):

        urls, embeddings = self.urls, np.vstack(self.article_embeddings)

        query_vec = vectorize_text(query, self.word_embeddings, self.dim_embed)
        # reshape for sklearn version 0.19
        query_vec = query_vec.reshape(1,-1)

        rec_urls = []

        for i in range(K):
            rec = pairwise_distances_argmin(query_vec, embeddings, metric='cosine')
            rec_urls.append(urls[rec][0])
            urls = np.delete(urls, rec)
            embeddings = np.delete(embeddings, rec, 0)

        return rec_urls

class ResponseManager(object):
    """ Manage all PoliBot responses """

    def __init__(self):

        # Intent recognition:
        # (restore TensorFlow graph and class instance from pre-trained model)
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(home + 'results/intent_classifier.meta')
        self.saver.restore(self.sess, home + 'results/intent_classifier')
        self.intent_model = unpickle(home + 'results/intents.pkl')
        self.input_batch = tf.get_collection('input_batch')[0]
        self.input_batch_len = tf.get_collection('input_batch_len')[0]
        print('Intents loaded.')

        # Article ranking system:
        self.article_ranker = ArticleRanker()
        print('Article ranker complete.')

        # Train conversational model
        self.create_dialogue()

    # Initialize self.dialogue with basic conversations trained on movie captions
    def create_dialogue(self):

        self.dialogue = ChatBot('thePoliBot',
            trainer='chatterbot.trainers.ChatterBotCorpusTrainer')

        self.dialogue.train('chatterbot.corpus.english')
        print('Conversations trained.')

    # Distinguish between general dialogue and political queries using intent classifier, then generate corresponding response to the user
    def create_response(self, query):

        # Display current source list
        if str.lower(query)=='sources':
            return "Here is your current source list: \n\n" + display_sources(self.article_ranker.source_map)

        # Prune specified sources and notify user
        if re.match('^[0-9]+(,[0-9]+)*$', re.sub(' ','',query)):
            removed_sources = prune_sources(query, self.article_ranker)
            if removed_sources=='VALUE_ERROR':
                return "Please choose only numbers between %d and %d." % (min(self.article_ranker.source_map), max(self.article_ranker.source_map))
            return "Alright, I'll stop recommending articles from " + removed_sources + "."

        # Classify user intent
        query_processed = process_text(query).split()
        q_input, q_input_len = input_batch_to_ids([query_processed], word_to_id)
        pred = self.sess.run('predictions:0',
                        feed_dict={self.input_batch: q_input,
                                   self.input_batch_len: q_input_len})
        intent = 'political' if pred>=0.5 else 'dialogue'

        # Political article recs response:
        if intent == 'political':
            # Feed processed query to article ranking system
            article_urls = self.article_ranker.choose_article_recs(query)
            return 'This sounds like a political question. You might be interested in these articles: \n\n' + '\n\n'.join(article_urls)

        # General dialogue response:
        else:
            # Feed query to general dialogue response component
            response = self.dialogue.get_response(query)
            return 'Sounds like you just want to chat. ' + str(response)
