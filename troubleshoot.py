# TROUBLESHOOTING CODE
home = '/Users/tracelevinson/Documents/DataScience/PoliBot/'

# import csv
# with open(home + 'data/raw_urls_test_corrected.tsv', 'w') as f:
#     reader = csv.reader(open(home+'data/raw_urls.tsv'))
#     for row in reader:
#         row_list = row[0].split('\t')[:2]
#         print(row_list)
#         if not re.search('bbc.co', row_list[0]):
#             f.write(str(row_list[0])+'\t'+str(row_list[1])+'\n')

# RESTORE TSV VALUE FROM CSV (ENABLE EDITING IN EXCEL)
# import pandas as pd
# home = '/Users/tracelevinson/Documents/DataScience/PoliBot/'
# articles = pd.read_csv(home + 'data/articles.tsv', sep='\t', header=0)
# articles = pd.read_csv(home + 'data/articles.csv', header=0)
# articles.to_csv(home + 'data/articles.tsv', sep='\t', header=True,  mode='w')

# REMOVE UNWANTED EMBEDDING VECTORS
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import pandas as pd
# embeddings = pd.read_csv(home+'results/ss_embeddings.tsv', encoding='utf-8', header=None)
# new_embeddings = pd.DataFrame(line.split('\t') for line in embeddings[0])
# new_embeddings[0] = new_embeddings[0].apply(lambda x: x.strip('.-,;:\"\' '))
# new_embeddings[100] = new_embeddings[100].apply(lambda x: x.strip('.'))
# new_embeddings[0] = new_embeddings[0].apply(lambda x: x if str.lower(x) not in stopwords.words('english') else '')
# new_embeddings[0].replace('[A-Za-z]+.* 0.[0-9]+','', regex=True, inplace=True)
# new_embeddings[100].replace('[^0-9.-]','', regex=True, inplace=True)
# new_embeddings = new_embeddings[~new_embeddings[100].str.contains('[\-]*0\.[0-9]+[^0-9]+.*$','[\-]*0\.[0-9]+', regex=True)]
# new_embeddings = new_embeddings.ix[:,0:100]
# new_embeddings = new_embeddings[new_embeddings[0]!='']
# new_embeddings = new_embeddings[new_embeddings[100]!='']
# new_embeddings.to_csv(home + 'results/ss_embeddings_new.tsv', sep='\t', header=False, index=False)

# NARROW DATASET TO ONE SOURCE
# # Limit to specific source temporarily
# articles_df = articles_df[articles_df.source=='cnn']
# articles_df.reset_index(inplace=True, drop=True)

# RESTORE TF SESSION TESTING
from intent_classifier import *
sess = tf.Session()
saver = tf.train.import_meta_graph(home + 'results/intent_classifier.meta')
# saver.restore(sess, home + 'results/intents.ckpt')
saver.restore(sess, home + 'results/intent_classifier') #tf.train.latest_checkpoint('./'))
model = unpickle(home + 'results/intents.pkl')
model.input_batch = tf.get_collection('input_batch')[0]
model.input_batch_len = tf.get_collection('input_batch_len')[0]
# graph = tf.get_default_graph()
# model.input_batch = graph.get_operation_by_name('input_batch')
# model.logits = graph.get_collection('logits')
# model.predictions = graph.get_collection('predictions')
query = "Why was Russia involved in the 2016 election?"
query_processed = process_text(query, intents=True).split()
query_x, query_x_len = input_batch_to_ids([query_processed], word_to_id)
print(sess.run('predictions:0', feed_dict={model.input_batch: query_x, model.input_batch_len: query_x_len}))

# model.set_placeholders()
# pred = model.query_predict(sess, query_x, query_x_len)

# MANUALLY UPDATE LOSS/ACC AND CONFUSION_MATRIX PLOTS
# all_train_losses = [0.696802, 0.365114, 0.366471, 0.356616, 0.360487, 0.310928, 0.297060, np.nan, 0.263141, 0.281445, 0.283057, 0.219097, 0.294121, 0.167343, np.nan, 0.232992, 0.191195, 0.252945, 0.126118, 0.223080, 0.159255, np.nan, 0.176448, 0.190235, 0.179750, 0.262535, 0.158613, 0.219362, np.nan, 0.139102, 0.139601, 0.156799, 0.188312, 0.146419, 0.156664, np.nan, 0.203837, 0.094464, 0.115000, 0.228585, 0.224053, 0.243757, np.nan, 0.269464, 0.147370, 0.202699, 0.155289, 0.157171, 0.172821, np.nan, 0.154455, 0.134221, 0.126788, 0.236606, 0.178301, 0.192852, np.nan, 0.120171, 0.082584, 0.167884, 0.135420, 0.122505, 0.197634, np.nan, 0.081920, 0.140007, 0.176358, 0.179549, 0.223791, 0.106306, np.nan]
#
# all_val_accs = [0.490611, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.896656, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.909363, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.917591, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.920837, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.925417, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.926348, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.927480, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.928839, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.929846, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.929720]
#
# all_steps = [0, 400, 800, 1200, 1600, 2000, 2400, 2486, 2886, 3286, 3686, 4086, 4486, 4886, 4972, 5372, 5772, 6172, 6572, 6972, 7372, 7458, 7858, 8258, 8658, 9058, 9458, 9858, 9944, 10344, 10744, 11144, 11544, 11944, 12344, 12430, 12830, 13230, 13630, 14030, 14430, 14830, 14916, 15316, 15716, 16116, 16516, 16916, 17316, 17404, 17804, 18204, 18604, 19004, 19404, 19804, 19888, 20288, 20688, 21088, 21488, 21888, 22288, 22374, 22774, 23174, 23574, 23974, 24374, 24774, 24860]
#
# test_acc = 0.929514

# test_truths = [0]*19245 + [1]*17732 + [0]*956 + [1]*1848
# test_preds = [0]*19245 + [1]*17732 + [1]*956 + [0]*1848
