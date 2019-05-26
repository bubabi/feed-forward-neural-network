import dynet_config
dynet_config.set_gpu()
import dynet as dy
import numpy as np


class NeuralNetwork:

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        self.model = dy.Model()
        self.trainer = dy.MomentumSGDTrainer(self.model, learning_rate=learning_rate)
        self.W = self.model.add_parameters((hidden_dim, input_dim))
        self.b_bias = self.model.add_parameters((hidden_dim,))
        self.U = self.model.add_parameters((output_dim, hidden_dim))
        self.d_bias = self.model.add_parameters((output_dim,))

    # Training the network
    def train(self, X, y):

        closs = 0
        #print(X[0])
        # Calculation of loss for each input
        for inp, out in zip(X, y):
            i_idx = np.argmax(inp)
            o_idx = np.argmax(out)
            print(i_idx, o_idx)
            dy.renew_cg()
            inp = dy.inputVector(inp)

            # f is the tanh activation function in the hidden layer
            f = dy.tanh(self.W * inp + self.b_bias)

            # Applying softmax and calculating loss
            loss = dy.pickneglogsoftmax(self.U * f + self.d_bias, o_idx)
            closs += loss.npvalue()
            loss.backward()
            self.trainer.update()

        print(closs/433)

    # Predict method calculates probability of P(Y=y | X) for all y values
    # def predict_proba(self, x):
    #     x = dy.inputVector(x)
    #     h = dy.rectify(self.W * x + self.b_bias)
    #     logits = self.U * h
    #
    #     # Converting outputs to probabilities by using softmax function
    #     temp = np.exp(logits.npvalue())
    #     prob_lst = temp / np.sum(temp)
    #     return prob_lst