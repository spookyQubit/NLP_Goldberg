
from collections import defaultdict
import dynet as dy
from random import shuffle
import numpy as np


"""
Data:
   X: movie review
   y: score given corresponding to the review
Purpose: For a given movie review, predict its corresponding score
Method:
Bag Of Words: We assume that each word is represented as a vector
              with length being the same as the number of different
              possible values for scores.
              For example, if the scores are: {0, 1, 2, 3, 4},
              with 0 being very-bad and 4 being very-good,
              then a word like "nice" can be associated with the following vector:
              [-3.2, -2.0, 0.2, 1.0, 0.0]
                |                     |
                |                     |
                V                     V
            Score for "nice"        Score for "nice"
            being very-bad          being very-good
"""


test_data_file_path = "../nn4nlp2017-code-master/data/classes/test.txt"
train_data_file_path = "../nn4nlp2017-code-master/data/classes/train.txt"

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]


def get_data(file_path):
    # Function to read in the data
    with open(file_path, 'r') as f:
        for line in f.readlines():
            score, review = line.lower().strip().split(" ||| ")
            yield (t2i[score], [w2i[word] for word in review.split(" ")])

train = list(get_data(train_data_file_path))
w2i = defaultdict(lambda: UNK, w2i)
test = list(get_data(test_data_file_path))
nTags = len(t2i)
nWords = len(w2i)


# Define the model parameters
model = dy.Model()
trainer = dy.AdamTrainer(model)

mW = model.add_lookup_parameters((nWords, nTags))
mb = model.add_parameters((nTags))


def calculate_scores_vector_for_list_of_words(word_idxs):
    dy.renew_cg()
    b = dy.parameter(mb)
    score_vector = dy.esum([dy.lookup(mW, x) for x in word_idxs])
    return b + score_vector


for ITER in range(100):
    shuffle(train)
    train_loss = 0.0
    for tag_idx, word_idxs in train:
        current_loss = dy.pickneglogsoftmax(calculate_scores_vector_for_list_of_words(word_idxs), tag_idx)
        train_loss += current_loss.value()
        current_loss.backward()
        trainer.update()
    print("iter %r: train loss/sent=%.4f" % (ITER, train_loss / len(train)))

    test_correct = 0.0
    for tag_idx, word_idxs in test:
        current_scores = calculate_scores_vector_for_list_of_words(word_idxs).npvalue()
        predict = np.argmax(current_scores)
        if predict == tag_idx:
            test_correct += 1.
    print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test)))



