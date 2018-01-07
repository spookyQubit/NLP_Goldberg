import dynet as dy
from collections import defaultdict
from random import shuffle
import math
import time

"""
Purpose: Create a language model incorporating neural-nets
Each word in the vocab is mapped into an embedding vector.

input = concatenate(emb(w_2), emb(w_1))
h1 = tanh(W1 * input + b1)
probability of the next word = softmax(W2*h1 + b2)
loss = sum of all cross entropy loss for each next word prediction
"""

# Trigram model
N = 2
EMB_SIZE = 64
H_SIZE = 64

valid_data_file_path = "../nn4nlp2017-code-master/data/ptb/test.txt"
train_data_file_path = "../nn4nlp2017-code-master/data/ptb/train.txt"

w2i = defaultdict(lambda: len(w2i))
S = w2i["<s>"]
UNK = w2i["<unk>"]


def read_dataset(path):
    with open(path, "r") as f:
        for line in f:
            yield [w2i[word] for word in line.strip().split(" ")]

train = list(read_dataset(train_data_file_path))
w2i = defaultdict(lambda: UNK, w2i)
valid = list(read_dataset(valid_data_file_path))
nWords = len(w2i)

# dynet model
model = dy.Model()
trainer = dy.SimpleSGDTrainer(model)

# parameters
W_emb_m = model.add_lookup_parameters((nWords, EMB_SIZE))
# layer 1
W1_m = model.add_parameters((H_SIZE, N * EMB_SIZE))
b1_m = model.add_parameters((H_SIZE))
#layer 2
W2_m = model.add_parameters((nWords, H_SIZE))
b2_m = model.add_parameters((nWords))


def get_history_score(hist):
    emb = dy.concatenate([dy.lookup(W_emb_m, hist_idx) for hist_idx in hist])
    W1 = dy.parameter(W1_m)
    b1 = dy.parameter(b1_m)
    h1 = dy.affine_transform([b1, W1, emb])
    W2 = dy.parameter(W2_m)
    b2 = dy.parameter(b2_m)
    return dy.affine_transform([b2, W2, h1])


def get_sentence_loss(sent):
    dy.renew_cg()
    hist = [S] * N
    all_scores = []
    for word_idx in sent:
        hist_score = get_history_score(hist)
        all_scores.append(dy.pickneglogsoftmax(hist_score, word_idx))
        hist = hist[1:] + [word_idx]
    return dy.esum(all_scores)


for ITER in range(100):
    shuffle(train)
    train_loss = 0
    train_words = 0
    start = time.time()
    for train_idx, sent in enumerate(train):
        sent_loss = get_sentence_loss(sent)
        train_loss += sent_loss.value()
        train_words += len(sent)
        sent_loss.backward()
        trainer.update()
        if (train_idx + 1) % 5000 == 0:
            print("--finished %r sentences" % (train_idx + 1))
            print("iter {0}: train loss/word={1}, ppl={2}, time={3}".format(ITER, train_loss / train_words, math.exp(train_loss / train_words), time.time() - start))
    print("Done training")
    valid_loss = 0
    valid_words = 0
    for valid_idx, sent in enumerate(valid):
        sent_loss = get_sentence_loss(sent)
        valid_loss += sent_loss.value()
        valid_words += len(sent)
        trainer.update()
        print("iter {0}: dev loss/word={1}, ppl={3}, time={3}".format(ITER, valid_loss / valid_words, math.exp(valid_loss / valid_words), time.time() - start))

