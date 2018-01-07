from collections import defaultdict
import dynet as dy
from random import shuffle
import numpy as np

"""
Data:
   X: movie review
   y: score given corresponding to the review
Purpose: For a given movie review, predict its corresponding score
Method: We assume that each word is represented as a vector
        with some predefined length EMB_SIZE.
        Unlike bow, here the EMB_SIZE need not be of the same
        length as the number of unique tags.
Model:  The following equation is used
        W * (sum of vectors of length EMB_SIZE, each for a given word in the sentence) + b
        cbow = (nWords, EMB_SIZE)
        W = (nTags, EMB_SIZE)
        b = (nTags)
"""


test_data_file_path = "../nn4nlp2017-code-master/data/classes/test.txt"
train_data_file_path = "../nn4nlp2017-code-master/data/classes/train.txt"


def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])


w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
train = list(read_dataset(train_data_file_path))
w2i = defaultdict(lambda: UNK, w2i)
test = list(read_dataset(test_data_file_path))
nWords = len(w2i)
nTags = len(t2i)

# Model parameters
EMB_SIZE = 64
model = dy.Model()
mCbow = model.add_lookup_parameters((nWords, EMB_SIZE))
mW = model.add_parameters((nTags, EMB_SIZE))
mb = model.add_parameters((nTags))
trainier = dy.AdamTrainer(model)


def calculate_sentence_emb(word_idxs):
    dy.renew_cg()
    W = dy.parameter(mW)
    b = dy.parameter(mb)
    emb = dy.esum([dy.lookup(mCbow, idx) for idx in word_idxs])
    return W * emb + b


for ITER in range(100):
    shuffle(train)
    train_loss = 0.0
    for word_idxs, tag_idx in train:
        emb = calculate_sentence_emb(word_idxs)
        loss = dy.pickneglogsoftmax(emb, tag_idx)
        train_loss += loss.value()
        loss.backward()
        trainier.update()
    print("iter %r: train loss/sent=%.4f" % (ITER, train_loss / len(train)))
    test_correct = 0.0
    for word_idxs, tag_idx in test:
        current_emb = calculate_sentence_emb(word_idxs).npvalue()
        predict = np.argmax(current_emb)
        if predict == tag_idx:
            test_correct += 1.
    print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test)))



