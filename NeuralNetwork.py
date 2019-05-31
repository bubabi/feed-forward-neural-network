import dynet_config
dynet_config.set_gpu()
import dynet as dy
import numpy as np


class NeuralNetwork:

    def __init__(self, i2w, inp_dim, hid_dim, out_dim):
        self.model = dy.Model()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.trainer = dy.SimpleSGDTrainer(self.model, learning_rate=0.05)
        self.W = self.model.add_parameters((hid_dim, inp_dim))
        self.b_bias = self.model.add_parameters((hid_dim,))
        self.U = self.model.add_parameters((out_dim, hid_dim))
        self.d_bias = self.model.add_parameters((out_dim,))
        self.i2w = i2w

    def save_model(self):
        self.model.save("dy.model1")

    def load_model(self):
        self.model = dy.ParameterCollection()
        self.W = self.model.add_parameters((self.hid_dim, self.inp_dim))
        self.b_bias = self.model.add_parameters((self.hid_dim,))
        self.U = self.model.add_parameters((self.out_dim, self.hid_dim))
        self.d_bias = self.model.add_parameters((self.out_dim,))
        self.model.populate("dy.model1")

    def predict_output(self, x):
        x_vector = dy.inputVector(x)

        f = dy.tanh(self.W * x_vector + self.b_bias)
        probs = dy.softmax(self.U * f + self.d_bias).npvalue()
        selection = np.random.choice(self.inp_dim, p=probs/probs.sum())

        return selection, probs[selection]

    def train(self, X, y):
        total_loss = 0

        for inp, out in zip(X, y):
            o_idx = np.argmax(out)

            dy.renew_cg() # create a new computation graph for each input
            inp = dy.inputVector(inp)
            f = dy.tanh(self.W * inp + self.b_bias)

            # calculate the softmax of probs then calculate negative log
            # which is equivalent to: dy.pick(-dy.log(dy.softmax(self.U * f + self.d_bias)), o_idx)
            loss = dy.pickneglogsoftmax(self.U * f + self.d_bias, o_idx)
            total_loss += loss.npvalue()

            # back-propagating then updating the parameters
            # using dy.SimpleSGDTrainer
            loss.backward()
            self.trainer.update()

        print(total_loss/len(X))
