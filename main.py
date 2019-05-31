import math
import dynet_config
from NeuralNetwork import NeuralNetwork
dynet_config.set_gpu()
import numpy as np
import json
from keras.utils import to_categorical

path = 'unim_poem.json'
vocab = ['<START>', '<END>', '<NEWL>']
input_idx = list()
output_idx = list()


def load_data(path, num_of_poems):
    with open(path) as f:
        data = json.load(f)
    return [datum['poem'] for datum in data][:num_of_poems]


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


raw_poems = load_data(path, num_of_poems=10)
poems = tokenize(raw_poems, vocab)

# Task 1: Build Feed-Forward Neural Network Language Model (FNN)

word2index = dict()
index2word = list()

# preparing the list to reach the word
# from the index number and vice versa.
for word in vocab:
    index2word.append(word)
    word2index[word] = len(word2index)

# distributing the words of the poem to the input
# and output idx lists in the form of bi-gram.
for poem in poems:
    poem.insert(0, "<START>")
    poem.append("<END>")
    for i in range(len(poem)-1):
        pre_word = poem[i]
        next_word = poem[i+1]
        input_idx.append(word2index[pre_word])
        output_idx.append(word2index[next_word])

# using the keras library to express words
# in the form of one-hot vector
oht_inputs = to_categorical(np.array(input_idx))
oht_outputs = to_categorical(np.array(output_idx))

vocab_size = len(oht_inputs[0])

# sending necessary parameters to create the
# Feed-Forward Neural Network Language model
model = NeuralNetwork(i2w=index2word, inp_dim=vocab_size,
                      hid_dim=512, out_dim=vocab_size)

# training the model with custom epoch size which is 50.
for i in range(50):
    print("ITER:", i)
    model.train(oht_inputs, oht_outputs)

model.save_model()
model.load_model()

# Task 2: Poem Generation


def calc_perplexity(probs):
    log_sum = 0
    for prob in probs:
        log_sum -= math.log(prob)
    return math.pow(2, log_sum / len(probs))


def generate_poem(start_word, num_of_lines):

    pre_idx = word2index[start_word]
    pre_idx_vector = oht_inputs[pre_idx]
    next_idx = None

    sentence_idx = [pre_idx]  # store indices of all generated words
    line_count = 0
    prob_list = list()

    # poem generation stops until the end of poem is reached or
    # the total number of lines is reached
    while next_idx != 1:

        next_idx, prob = model.predict_output(pre_idx_vector)
        prob_list.append(prob)

        if index2word[next_idx] == "<NEWL>":
            line_count += 1

        if line_count == num_of_lines:
            break

        sentence_idx.append(next_idx)
        pre_idx_vector = oht_inputs[next_idx]

    # calculate perplexity of the generated poem then print it
    perplex = calc_perplexity(prob_list)
    print("PERPLEXITY:", perplex)

    # print the generated poem
    for word_id in sentence_idx:
        word = index2word[word_id]

        if word == "<NEWL>": print("<NEWL>")
        else: print(word, end=" ")
    print("")


for i in range(5):
    generate_poem(start_word="<START>", num_of_lines=2)
    print("")
