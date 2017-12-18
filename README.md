# Language Modeling with Matrix Embedding
`lm_me` for short, uses (square, optionally sparse) matrices as words and a sentence is the product of its words.

## Library
 * python 3
 * numpy, theano
 * 64 bit (double) precision may be advised.
   * see https://github.com/Theano/libgpuarray
   * or https://gist.github.com/gaebor/61323e72a7b415e6487e9590ce181857

## Features
### lm_me
* given a list of sentences (input or corpus file) and a dictionary (vocabulary) of tokens the tool learns matrices for each token in the vocabulary.
* the learned model is a generative probability model. For each list of tokens (sentence) you can get the estimated probability of that sentence.
* the objective function is the KL divergence from the gold probabilities of the input sentences. Or if you don't have probabilities, then simply the entropy of the generative model (log perplexity). In either case, it is measured in bits.
* separator or delimiter can be added between the token in the sentences, space by default.
* the training corpus is read batch-by-batch, it uses only as much memory as needed for a single batch. But optionally you can load the entire data into the memory if you want.
  * _Shake before use!_ Due to SGD, you can shuffle the sentences in the input data between epochs, for that you have to load the dataset at once. Otherwise you should shuffle the dataset offline, before you run `lm_me`.
* Run with the `-h` or `--help` flags for details on the parameters.

### babble
* You can randomly or deterministically generate sentences using a trained model.
* See `babble.py`
* This script loads a model and ends your sentences. If you type an interrupted sentence (or an empty one) then the script generates the end of it.
* You have to give an end-of-sentence symbol to know where to stop.
