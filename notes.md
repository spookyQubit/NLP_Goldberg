Chapter 1

### Chapter 6

   * TFIDF
   * How can one use the lixical information in the framework on NN. This will be discussed in chaper 11.
   * biRNN provides a framework to train for a flexible, **adjustable** and trainiable network.
   * MLP cannot infer "the bigram XY appears in the document"
   * biRNN and convolutional neural nets are used to extract ngram features
   * If Paris is present, the document is most likely a Travel document. If the word Hilton is present, it is again most likely a Travel document. But if Paris and Hilton both appear in the document, it is most likely a Celebrity Gossip document. This is an example of XOR which is impossible to be modeled bya  linear model. This is where NN can be useful. 

### Chapter 7
   * NER: John Smith , president of McCormik Industries visited his niece Paris in Milan , reporters say. 
and the expected output would be:
[PER John Smith ] , president of [ORG McCormik Industries ] visited his niece [PER Paris ] in [LOC Milan ], reporters say .
   * While NER is a sequence segmentation task—it assigns labeled brackets over nonoverlapping sentence spans—it is often modeled as a sequence tagging task, like POS-tagging.  

### Chapter 8
   * When the core features are the words in a 5 words window surrounding and
including a target word (2 words to each side) with positional information, and a vocabulary of 40,000 words, x will be a 200,000 dimensional vector with 5 non-zero entries.
   * Each core feature is embedded into a d dimensional space, and represented as a vector in that space. Different feature types may be embedded into different spaces. For example, one may represent word features using 100 dimensions, and part-of-speech features using 20 dimensions.
   * The embeddings (the vector representation of each core feature) are treated as parameters of the network, and are trained like the other parameters.
   * Benifits of using one-hot encoding: Dimensionality of one-hot vector is same as number of distinct features. Features are completely independent from one another. e feature “word is ‘dog’ ” is as dissimilar to “word is ‘thinking’ ” than it is to “word is ‘cat’ ”. In cases where we have relatively few distinct features in the category, and we believe there are no correlations between the different features, we may use the one-hot representation.  
   * Benifits of using embeddings: Dimensionality of each vector is d. Model training will cause similar features to have similar vectors—information is shared between similar features.
   * CBOW: The (dense) vectors, each corresponding to a feature, is summed. This has the benifit that even if the number of featuers is not known at the beginning of the task, the input size is fixed. Note that if each feature is mapped to a one-hot-vevctor, CBOW reduces to BOW.
   * Distance and position features: Consider the coreference-resolution task of figuring out which candidate word corresponds to a pronoun. In such a task, the distance between the candidate word and the pronoun is an important feature. The question is how does one encode this feature? In the "traditional" NLP setup, the distances are encoded by binning the distance together (1, 2, 3, 4-10, 10+) and associating each bin with a one-hot-vector. In NN architectures, distance features are encoded similarly to the other feature types: each bin is associated with a d-dimensional vector, and these distance-embedding vectors are then trained as regular parameters in the network.
   * Padding: From what I understand, this is similar to adding a beginning/end of sentence symbol and let the NN learn a representation for these symbols.
   * Unknown words: Use <unk> to represent a word which you have not encountered while training.
   * Backing off techniques: we may replace unknown words that end with ing with an *__ing* symbol, and so on. These are generally hand crafted.
   * Word droput: In order to learn a good representation of <unk>, less frequent words can be dropped from training and replaced with <unk>. The other alternative is to randomly replace words by <unk>. The words which are replaced by <unk> in one iteration may not be replaced by <unk> in the next iteration.
   * Word dropout can be used as a regularization technique where a word is dropped according to a Bernoulli trail probability, p.
   * Comparison with kernel methods: Kernel methods and in particular polynomial kernels, allow the feature designer to specify only core features, leaving the feature combination aspect to the learning algorithm. In contrast to neural network models, kernels methods are convex, admitting exact solutions to the
optimization problem. However, the computational complexity of classification in kernel methods scales linearly with the size of the training data, making them too slow for most practical purposes, and not suitable for training with large datasets. On the other hand, the computational complexity of classification using neural networks scales linearly with the size of the network, regardless of the training data size. Of course, one still needs to go over the entire dataset when training, and sometimes go over the dataset several times. This makes training time scale linearly with the dataset size. However, each example, in either training or test time, is processed in a constant time (for a given network). This is in contrast to a kernel classifier, in which each example is processed in a time that scales linearly with the dataset size.  
   * Book on kernel methods:  Kernel methods for pattern analysis
   * Perplexity: Check the exact definition of perplexity. How does one take into account the number of words and number fo sentences in normalization? Check: http://www.cs.columbia.edu/~mcollins/lm-spring2013.pdf
   * Kneser-Ney smoothing? 
   * Hierarchical softmax: Helps in reducing the computation time of the softmax layer. If one is interested in knowing only the probability of a word, and not of the entire vocabulary, then we can use hierarchical softmax. 


## Neural Summarization by Extracting Sentences and Words
A  few  recent  studies  (Kobayashi 2015; Yogatama 2015) perform sentence extraction based on pre-trained sentence embeddings following an unsupervised optimization paradigm. Our work also uses continuous representations to express the meaning of sentences and documents, but importantly employs neural networks more di-
rectly to perform the actual summarization task.
## Chaper 1
```
SGD Techniques
 * SGD+Momentum
 * Nestrov Momentum
 * AdaGrad
 * AdaDelta
 * RMSProp
 * Delta
```
