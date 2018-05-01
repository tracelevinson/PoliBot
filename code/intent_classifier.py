"""
intent_classifier.py

Intent classification model to decipher between political and general dialogue intents.

Input: intents.tsv (pre-processed intents dataset from intents_prep.py)
Outputs: (1) intents.pkl (IntentClassifier class instance with trained model weights/embeddings)
         (2) intent_classifier.meta (saved TensorFlow meta graph for importation in final bot use)
"""
###
from utils import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from numpy import random
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sn
###

intents = pd.read_csv(home + 'data/intents.tsv', encoding = 'ISO-8859-1', sep='\t', header=0)

# Convert intents to booleans (True for political, False for dialogue)
intents.intent = (intents.intent == 'political')

# Split data into training (80%), validation (10%) and test (10%) sets
train_val, test = train_test_split(intents.as_matrix(), test_size=0.1, random_state=0)
train, val = train_test_split(train_val, test_size=0.111, random_state=0)

# Create dictionaries mapping between vocab and numerical IDs
word_to_id = {word:id for id, word in enumerate(sorted(set(intents.text.str.cat(sep=' ').split())))}
word_to_id['#'] = len(word_to_id) # padding symbol for shorter sentences

# Save word_to_id to pickle for AWS memory efficiency.
pickle.dump(word_to_id, open(home+'results/word_to_id.pkl', 'wb'), -1)

id_to_word = {id:word for word, id in word_to_id.items()}

# Create sentence ID vectors of length max_len (longest sentence in sample). Pad shorter sentences with '#' to standardize within-batch lengths for efficiency.
def sentence_to_ids(text, word_to_id, max_len):

    sent_ids = ([word_to_id[w] if w in word_to_id else word_to_id['#'] for w in text][0:max_len] +
                [word_to_id['#']] * max(0,max_len-len(text)))
    sent_len = min(max_len,len(text))

    return sent_ids, sent_len

# Convert sentence from IDs back to words
def ids_to_sentence(ids, id_to_word):

    return [id_to_word[i] for i in ids]

# Convert full input batch of sentences to IDs
def input_batch_to_ids(sentences, word_to_id):

    max_batch_len = max(len(s) for s in sentences)
    batch_ids, batch_lens = [], []
    for sent in sentences:
        ids, sent_len = sentence_to_ids(sent, word_to_id, max_batch_len)
        batch_ids.append(ids)
        batch_lens.append(sent_len)
    return batch_ids, batch_lens

# Generate batches from observation samples
def make_batches(samples, batch_size=128):
    sentences, intents = [], []
    for i, (sent, intent) in enumerate(samples, 1):
        sentences.append(sent.split())
        intents.append(intent)
        if i % batch_size == 0:
            yield sentences, intents
            sentences, intents = [], []
    if sentences and intents:
        yield sentences, intents

#################################
# SET UP MODEL, INPUTS & OUTPUTS
#################################

# Establish TensorFlow class for intent classifier
class IntentClassifier(object):
    pass

# Create model placeholders
def set_placeholders(self):

    # Input batches
    self.input_batch = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_batch')
    self.input_batch_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='batch_lengths')

    # Ground truths ('dialogue' or 'political')
    self.ground_truth = tf.placeholder(shape=(None,), dtype=tf.float32, name='ground_truth')

    # Additional variables
    self.keep_prob = tf.placeholder_with_default(1.0, shape = [], name='keep_prob')
    self.learning_rate = tf.placeholder(shape=(), dtype=tf.float32, name='learning_rate')

IntentClassifier.set_placeholders = classmethod(set_placeholders)

# Create input batch embeddings for LSTM
def make_embeddings(self, vocab_size, embed_size):

    # Embedding layer variable
    self.embeddings = tf.Variable(
                      tf.random_uniform((vocab_size, embed_size), -1.0, 1.0),
                      dtype=tf.float32,
                      name='embeddings')

    # Look up embeddings for input batch
    self.input_batch_embed = tf.nn.embedding_lookup(self.embeddings, self.input_batch,
                                                    name='input_batch_embed')

IntentClassifier.make_embeddings = classmethod(make_embeddings)

# Build out bidirectional LSTM model
def build_lstm(self, layer_size):

    # Create forward LSTM cell with dropout
    core_forward =  tf.nn.rnn_cell.LSTMCell(layer_size,
                                            forget_bias=1.0,
                                            reuse=tf.get_variable_scope().reuse)
    forward_cell = tf.nn.rnn_cell.DropoutWrapper(core_forward,
                                            input_keep_prob = self.keep_prob,
                                            dtype = tf.float32)

    # Create backward LSTM cell with dropout
    core_backward = tf.nn.rnn_cell.LSTMCell(layer_size,
                                            forget_bias=1.0,
                                            reuse=tf.get_variable_scope().reuse)
    backward_cell = tf.nn.rnn_cell.DropoutWrapper(core_backward,
                                            input_keep_prob = self.keep_prob,
                                            dtype = tf.float32)

    # Create Dynamic RNN
    _, output_states = tf.nn.bidirectional_dynamic_rnn(
                                            cell_fw=forward_cell,
                                            cell_bw=backward_cell,
                                            inputs=self.input_batch_embed,
                                            sequence_length=self.input_batch_len,
                                            dtype=tf.float32)

    forward_state = output_states[0]
    backward_state = output_states[1]
    final_state = tf.concat([forward_state.h, backward_state.h], axis=1)

    self.logits = tf.squeeze(
                  tf.contrib.layers.fully_connected(inputs=final_state,
                                                    num_outputs=1,
                                                    activation_fn=None,),
                  name='logits')

IntentClassifier.build_lstm = classmethod(build_lstm)


# Compute sigmoid cross entropy loss on RNN output
def compute_loss(self):

    self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.ground_truth,
                                                logits=self.logits)

IntentClassifier.compute_loss = classmethod(compute_loss)

# Perform Adam optimization on loss function
def optimize_loss(self):

    self.train_op = tf.contrib.layers.optimize_loss(loss = self.loss,
                                                    global_step = tf.train.get_global_step(),
                                                    learning_rate = self.learning_rate,
                                                    optimizer = 'Adam',
                                                    clip_gradients = 1.0,
                                                    name='train_op')

IntentClassifier.optimize_loss = classmethod(optimize_loss)

################################
# SET UP MODEL RUN & EVALUATION
################################

# Initialize classification model
def model_init(self, vocab_size, embed_size, layer_size):

    # Set up model
    self.set_placeholders()
    self.make_embeddings(vocab_size, embed_size)
    self.build_lstm(layer_size)

    # Perform loss computation and optimization
    self.compute_loss()
    self.optimize_loss()

    # Obtain class predictions
    self.predictions = tf.nn.sigmoid(self.logits, name='predictions')

IntentClassifier.__init__ = classmethod(model_init)

# Train model on input batch
def batch_train(self, session, X, X_len, Y, learning_rate, dropout_keep_prob):
    feed_dict = {
            self.input_batch: X,
            self.input_batch_len: X_len,
            self.ground_truth: Y,
            self.learning_rate: learning_rate,
            self.keep_prob: dropout_keep_prob
        }
    pred, loss, _ = session.run([
            self.predictions,
            self.loss,
            self.train_op], feed_dict=feed_dict)
    return pred, loss

IntentClassifier.batch_train = classmethod(batch_train)

# Return predictions and loss for input batch
def batch_predict(self, session, X, X_len, Y):
    feed_dict = {
            self.input_batch: X,
            self.input_batch_len: X_len,
            self.ground_truth: Y
        }
    pred, loss = session.run([
            self.predictions,
            self.loss,
            ], feed_dict=feed_dict)
    return pred, loss

IntentClassifier.batch_predict = classmethod(batch_predict)

##################################
# INSTANTIATE & RUN CURRENT MODEL
##################################

if __name__=='__main__':

    tf.reset_default_graph()

    # Establish model instance and hyperparameters
    model = IntentClassifier(vocab_size = len(word_to_id),
                             embed_size = 50,
                             layer_size = 512)
    batch_size = 128
    n_epochs = 10
    learning_rate = 0.0005
    dropout_keep_prob = 0.5
    steps = int(len(train) / batch_size)

    # Initialize session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # Store global arrays for visualizations/evaluation
    all_val_preds    = []
    all_val_truths   = []
    all_train_losses = []
    all_val_accs     = []
    all_steps        = []

    ##################################################
    # TRAIN MODEL AND RETAIN VALIDATION LOSS/ACCURACY
    ##################################################

    print('Begin training: \n')
    for epoch in range(1, n_epochs+1):
        random.shuffle(train)
        random.shuffle(val)

        print('Training epoch', epoch)
        for step, (X_batch, Y) in enumerate(make_batches(train, batch_size=batch_size)):
            # Prepare training batch
            X, X_len = input_batch_to_ids(X_batch, word_to_id)
            preds, loss = model.batch_train(session, X, X_len, Y, learning_rate, dropout_keep_prob)

            if step % 400 == 0 and (epoch==1 or step>0):
            # if step % 10 == 0:
                all_train_losses.append(loss)
                all_val_accs.append(np.nan)
                all_steps.append(step + (epoch-1)*steps)
                print("Epoch: [%d/%d], step: [%d/%d], loss: %f" % (epoch, n_epochs, step+1, steps, loss))
        X_sent, Y = next(make_batches(val, batch_size=batch_size))

        # prepare val data (X_sent and Y_sent) for loss predictions
        X, X_len = input_batch_to_ids(X_sent, word_to_id)
        preds, loss = model.batch_predict(session, X, X_len, Y)
        print('Validation epoch', epoch, 'loss:', loss,'\n')
        for x, y, p  in list(zip(X, Y, preds))[:3]:
            print('X:',' '.join(ids_to_sentence(x, id_to_word)))
            print('Y:',str(y))
            print('P:',str(p),'\n')

        val_preds, val_truths = [], []
        # Compute ground truths, predictions, and accuracy over all val obs.
        for X_batch, Y in make_batches(val, batch_size=batch_size):

            X, X_len = input_batch_to_ids(X_batch, word_to_id)
            preds, _ = model.batch_predict(session, X, X_len, Y)

            for y, p in list(zip(Y, preds)):
                val_truths.append(int(y))
                val_preds.append(int(p>=0.5))

        all_val_preds.append(val_preds)
        all_val_truths.append(val_truths)

    print('\nTraining complete.')

    ######################################
    # EVALUATE VALIDATION & TEST ACCURACY
    ######################################

    # Evaluate val accuracy over each epoch
    for i, (truths, preds) in enumerate(zip(all_val_truths, all_val_preds), 1):
        val_acc = accuracy_score(truths, preds)
        all_val_accs.append(val_acc)
        all_train_losses.append(np.nan)
        all_steps.append(i*steps)
        print("Epoch: %i, Validation Accuracy: %f" % (i, val_acc))

    # Evaluate final test accuracy
    test_truths, test_preds = [], []
    for X_batch, Y in make_batches(test, batch_size=batch_size):

        X, X_len = input_batch_to_ids(X_batch, word_to_id)
        preds, _ = model.batch_predict(session, X, X_len, Y)

        for y, p in list(zip(Y, preds)):
            test_truths.append(int(y))
            test_preds.append(int(p>=0.5))

    test_acc = accuracy_score(test_truths, test_preds)
    print("Test Accuracy: %f" % test_acc)

    #####################################################
    # PLOT LOSS/ACCURACY OVER EPOCHS & CONFUSION MATRIX
    #####################################################

    # Loss/accuracy plot
    viz_df = pd.DataFrame({'Train Loss': all_train_losses, 'Val Accuracy': all_val_accs},
                            index=[x/steps for x in all_steps]).sort_index().interpolate()
    viz_df.plot(color=['red','grey'], lw=0.5)
    plt.fill_between(viz_df.index, viz_df['Train Loss'], viz_df['Val Accuracy'], facecolor='lightgreen', alpha=0.5)
    plt.plot(viz_df.index[-1], test_acc, marker='D', markersize=3, color='blue', linestyle='None', label='Test Accuracy')
    plt.title('LSTM Loss & Accuracy Evolution')
    plt.xlabel('Epoch')
    plt.legend(loc='lower left', fontsize='small', framealpha=1)
    plt.xlim(0, viz_df.index[-1]+0.5)
    plt.ylim(-0.1, 1)
    plt.axhline(y=0, color='black', lw=0.5, linestyle='--', dashes=(5,10))
    plt.savefig('../images/loss_accuracy.png')

    # Confusion matrix
    plt.figure()
    cm = confusion_matrix(test_truths, test_preds)
    sn.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Testing Confusion Matrix')
    plt.savefig('../images/confusion_matrix.png')

    #############
    # SAVE MODEL
    #############

    # Save intent classifier model and TensorFlow graph for bot use
    pickle.dump(model, open(home + 'results/intents.pkl', 'wb'), -1)
    tf.add_to_collection('input_batch', model.input_batch)
    tf.add_to_collection('input_batch_len', model.input_batch_len)
    saver = tf.train.Saver()
    saver.save(session, home + 'results/intent_classifier')
