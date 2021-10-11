
#===============================================================================
# process_text.py written by lucius-verus-fan 2021-10-10
#
# Tokenizes war-and-peace.txt and constructs word index arrays.
# Inspired by pangolulu/rnn-from-scratch/preprocessing.py.
#===============================================================================

import nltk
import itertools
import numpy as np

def process_text():

    # Set vocabulary size.
    vocabulary_size = 10000

    # Read in .txt and split into sentences.
    with open ('war-and-peace.txt') as txtfile:
        sentences = nltk.sent_tokenize(txtfile.read())

    # Convert sentences to lowercase.
    sentences = [sentences.lower() for sentences in sentences]

    # Append sentence tags.
    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

    # Remove punctuation, tokenize, filter sentences with less than 8 words.
    punctuation = nltk.RegexpTokenizer(r"\w+")
    tokenized_sentences = [punctuation.tokenize(sentences) for sentences in sentences]
    tokenized_sentences = list(filter(lambda x: len(x) >= 8, tokenized_sentences))

    # Count word frequency and create word index.
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i, w in enumerate(index_to_word)])

    # Replace words not in vocab with unknown token.
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    # Create training array, y_train is lagged by one step for prediction task.
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    return X_train, Y_train

