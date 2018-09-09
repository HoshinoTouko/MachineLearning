'''
File: BPNN.py
Created Date: 08 Sep, 2018
Author: Touko Hoshino
-----
Copyright (c) 2018 Hoshino Touko
'''
from sklearn.model_selection import train_test_split
import numpy as np

from data.fake.bayes import generate_bayes_fake_data_for_test


sigmoid = lambda x: 1 / (1 + np.exp(-x))
np_sigmoid = np.vectorize(sigmoid)


class BPNN:
    def __init__(self, features, labels, input_layer, hidden_layer, output_layer):
        self.num_of_data = len(features)
        self.features = features
        self.labels = labels
        self.NN = SingleHiddenLayerBPNNArchitecture(input_layer, hidden_layer, output_layer)
    
    def train(self, times=100, break_acc=0.9):
        for _time in range(times):
            self.NN.train(
                self.features[_time % self.num_of_data], 
                self.labels[_time % self.num_of_data]
            )
            if _time % 700 == 0:
                acc = self.analysis()
                print('Trained %d times, get an accurate of %.2f%%' % (_time, 100 * acc))
                if acc > break_acc:
                    break
    
    def analysis(self, features=None, labels=None):
        if features is None:
            features = self.features
        if labels is None:
            labels = self.labels
        
        total = len(features)
        acc = 0.
        for test_num in range(len(features)):
            feature = features[test_num]
            res = self.NN.predict(feature)
            for i in range(len(res)):
                if int(res[i]) != int(labels[test_num][i]):
                    break
            else:
                acc += 1
        return acc / total


class SingleHiddenLayerBPNNArchitecture:

    def __init__(self, input_layer, hidden_layer, output_layer):
        # Init limit
        self.l1 = 0.05
        self.l2 = 0.2
        self.times = 0
        # Init layers
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        self.threshold_hidden = np.random.uniform(-1, 1, (hidden_layer, ))
        self.threshold_output = np.random.uniform(-1, 1, (output_layer, ))

        self.mat_input_hidden = np.asmatrix(np.random.uniform(-1, 1, (input_layer, hidden_layer)))
        self.mat_hidden_output = np.asmatrix(np.random.uniform(-1, 1, (hidden_layer, output_layer)))
    
    def predict(self, feature):
        _, _output_layer = self._predict(feature)
        _output_layer = np.array(_output_layer).flatten()
        return list(map(np.round, _output_layer))


    def _predict(self, feature):
        feature = np.asmatrix(feature)
        _hidden_layer = np_sigmoid(feature * self.mat_input_hidden - self.threshold_hidden)
        _output_layer = np_sigmoid(_hidden_layer * self.mat_hidden_output - self.threshold_output)
        return _hidden_layer, _output_layer
        

    def train(self, feature, label):
        label = np.array(label)
        _hidden_layer, _output_layer = self._predict(feature)
        _hidden_layer = np.array(_hidden_layer).flatten()
        _output_layer = np.array(_output_layer).flatten()
        # Calculate gjs
        _gjs = _output_layer * (np.ones(_output_layer.shape) - _output_layer) * (label - _output_layer)
        # print(_gjs)
        # Calculate ehs
        _bhs = _hidden_layer * (np.ones(_hidden_layer.shape) - _hidden_layer)
        _ehs = _bhs * np.array(self.mat_hidden_output * np.asmatrix(_gjs).T).flatten()
        # print(_ehs)
        # Update w
        _d_w = self.l1 * (np.asmatrix(_bhs).T * np.asmatrix(_gjs))
        self.mat_hidden_output += _d_w
        # Update output threshold
        self.threshold_output -= self.l1 * _gjs
        # Update v
        _d_v = self.l2 * (np.asmatrix(feature).T * np.asmatrix(_ehs))
        self.mat_input_hidden += _d_v
        # Update hidden threshold
        self.threshold_hidden -= self.l2 * _ehs

        self.times += 1


    def show(self):
        print(self.mat_input_hidden)
        print(self.mat_hidden_output)


def pre_process(features):
    res = []
    for x in range(0, 16, 2):
        for y in range(0, 16, 2):
            tmp = features[x + y * 8]
            tmp += features[x + 1 + y * 8]
            tmp += features[x + (y + 1) * 8]
            tmp += features[x + 1 + (y + 1) * 8]
            res.append(tmp/4)
    return res


def main():
    print('Loading dataset...')
    with open('data/some_datasets/uspst_uni.txt') as fi:
        dataset = fi.readlines()
    with open('data/some_datasets/uspst_uni_label.txt') as fi:
        labelset = fi.readlines()
    features = []
    labels = []

    print('Pre processing...')
    for i in range(len(dataset)):
        line = dataset[i]
        data = list(map(float, line.split()))
        features.append(data)
        # features.append(pre_process(data))

        label = list(('0000' + str(bin(int(labelset[i]))).replace('0b', ''))[-4:])
        labels.append(list(map(int, label)))
    
    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.7)

    # print(features)
    # print(labels)
    print('Training...')
    bpnn = BPNN(x_train, y_train, 256, 30, 4)
    bpnn.train(99999, break_acc=0.99)
    # Test
    print('Testing...')
    print('Test result: %.4f%%' % (bpnn.analysis(x_test, y_test) * 100))

    # data, _ = generate_bayes_fake_data_for_test(times=100, discrete=True)
    # features = list(map(lambda x: x[0], data))
    # labels = list(map(lambda x: [x[1], 0], data))
    
    # bpnn = BPNN(features, labels, 4, 5, 2)
    # bpnn.train(500)

    # # bpnn.show()


if __name__ == '__main__':
    main()
