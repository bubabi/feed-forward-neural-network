import itertools

import dynet_config

from NeuralNetwork import NeuralNetwork

dynet_config.set_gpu()
import dynet as dy
import numpy as np
import json
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
import nltk

path = 'unim_poem.json'
UNK_TOKEN = "<UNK>"
min_count = 2


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return [datum['poem'] for datum in data][:100]


def tokenize(poems):
    tokens_lst = list()
    for i in range(len(poems)):
        poems[i] = nltk.word_tokenize(poems[i])
        for word in poems[i]:
            tokens_lst.append(word)
    return tokens_lst


poems = load_data(path)
tokens = tokenize(poems)

word2index = {'<START>': 0, '<END>': 1}
index2word = ["<START", "<END>"]

counter = Counter(tokens)
for word, count in counter.items():
    if count >= min_count:
        index2word.append(word)
        word2index[word] = len(word2index)


input_idx = list()
output_idx = list()

for poem in poems:
    poem.insert(0, "<START>")
    poem.append("<END>")
    print(poem)
    for i in range(len(poem)-1):
        pre_word = poem[i]
        next_word = poem[i+1]
        if {pre_word, next_word}.issubset(word2index):
            # print(pre_word, word2index[pre_word], next_word, word2index[next_word])
            input_idx.append(word2index[pre_word])
            output_idx.append(word2index[next_word])

    #print(input_words)
    #print(ending_words)
    print("")

inputs = np.array(input_idx)
outputs = np.array(output_idx)

oht_inputs = to_categorical(inputs)
oht_outputs = to_categorical(outputs)

# print(len(oht_inputs[0]), len(oht_outputs))
# print(oht_inputs)

# print("")

# print(oht_outputs)
# print(outputs)
# print(oht_outputs)
vocab_size = len(oht_inputs[0])
print(vocab_size)
model = NeuralNetwork(input_dim=vocab_size, hidden_dim=512,
                      output_dim=vocab_size, learning_rate=0.005)

for i in range(1):
    model.train(oht_inputs, oht_outputs)