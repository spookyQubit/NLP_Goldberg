import dynet as dy
from collections import defaultdict
from random import shuffle

"""
Purpose: Create a log-linear language model. We use a penn tree bank data set.
For each word, we create a lookup.
The model we use is as follows:
probability of next word = softmax(lookup(w_2) + lookup(w_1) + bias)
where
w_2 = 2 words to the left of the next word
w_2 = 1 word to the left of the next word
"""

# Trigram model
N = 2

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

# Dynet model parameters
model = dy.Model()
W_m = [model.add_lookup_parameters((nWords, nWords)) for _ in range(N)]
b_m = model.add_parameters((nWords))
trainer = dy.SimpleSGDTrainer(model)


def get_score_for_hist(hist):
    scores = [dy.parameter(b_m)]
    for h_, W_ in zip(hist, W_m):
        scores.append(dy.lookup(W_, h_))
    return dy.esum(scores)


def get_loss_for_sentence(sent):
    dy.renew_cg()
    hist = [S] * N
    loss = []
    for next_word in sent:
        score_for_hist = get_score_for_hist(hist)
        loss.append(dy.pickneglogsoftmax(score_for_hist, next_word))
        hist = hist[1:] + [next_word]
    return dy.esum(loss)

for ITER in range(1):
    shuffle(train)
    for sent_id, sent in enumerate(train):
        current_loss = get_loss_for_sentence(sent)
        current_loss.backward()
        trainer.update()
        if (sent_id + 1) % 5000 == 0:
            print("--finished %r sentences" % (sent_id + 1))
    # evaluate on dev set
    valid_loss = 0
    dev_words = 0
    for sent_id, sent in enumerate(valid):
        valid_loss += get_loss_for_sentence(sent).value()
        dev_words += len(sent)
        trainer.update()
        print("iter %r: dev loss/word=%.4f", (ITER, valid_loss / dev_words))




