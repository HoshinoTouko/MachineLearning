'''
File: BPNN.py
Created Date: 08 Sep, 2018
Author: Touko Hoshino
-----
Copyright (c) 2018 Hoshino Touko
'''
import numpy as np


sigmoid = lambda x: 1 / (1 + np.exp(-x))
np_sigmoid = np.vectorize(sigmoid)


class SingleHiddenLayerBPNNArchitecture:
    def __init__(self, input_layer, hidden_layer, output_layer):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        # self.threshold_hidden = np.zeros((hidden_layer, ))
        # self.threshold_output = np.zeros((output_layer, ))

        self.mat_input_hidden = np.asmatrix(np.random.uniform(-1, 1, (input_layer, hidden_layer)))
        self.mat_hidden_output = np.asmatrix(np.random.uniform(-1, 1, (hidden_layer, output_layer)))
    

    def predict(self, feature):
        feature = np.asmatrix(feature)
        _hidden_layer = np_sigmoid(feature * self.mat_input_hidden)
        print(_hidden_layer)
        _output_layer = np_sigmoid(_hidden_layer * self.mat_hidden_output)
        _output_layer = np.array(_output_layer).flatten()
        print(_output_layer)
        return list(map(np.round, _output_layer))


    def train(self, feature, label):
        pass
    

    def show(self):
        print(self.mat_input_hidden)
        print(self.mat_hidden_output)


def main():
    bpnn = SingleHiddenLayerBPNNArchitecture(3, 4, 5)
    print(bpnn.predict([1, 2, 3]))
    # bpnn.show()


if __name__ == '__main__':
    main()
