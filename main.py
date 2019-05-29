import itertools

import dynet_config
from NeuralNetwork import NeuralNetwork
dynet_config.set_gpu()
import dynet as dy
import numpy as np
import json
from keras.utils import to_categorical

path = 'unim_poem.json'
vocab = ['<START>', '<END>', '<NEWL>']
input_idx = list()
output_idx = list()


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return [datum['poem'] for datum in data][:100]


def tokenize(r_poems, vocab):
    container = list()
    for i in range(len(r_poems)):
        temp_poem = list()
        lines = r_poems[i].splitlines()
        for line in lines:
            for word in line.split():
                temp_poem.append(word)
                if not vocab.__contains__(word):
                    vocab.append(word)
            if not line == lines[-1]:
                temp_poem.append("<NEWL>")
        container.append(temp_poem)
    return container


raw_poems = load_data(path)
poems = tokenize(raw_poems, vocab)

word2index = dict()
index2word = list()

for word in vocab:
    index2word.append(word)
    word2index[word] = len(word2index)

for poem in poems:
    poem.insert(0, "<START>")
    poem.append("<END>")
    for i in range(len(poem)-1):
        pre_word = poem[i]
        next_word = poem[i+1]
        input_idx.append(word2index[pre_word])
        output_idx.append(word2index[next_word])

oht_inputs = to_categorical(np.array(input_idx))
oht_outputs = to_categorical(np.array(output_idx))

vocab_size = len(oht_inputs[0])

model = NeuralNetwork(i2w=index2word, inp_dim=vocab_size,
                      hid_dim=64, out_dim=vocab_size)

for i in range(50):
    model.train(oht_inputs, oht_outputs)

model.save_model()

# model.load_model()
model.predict_output(oht_inputs[word2index['<START>']])



