import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from functools import reduce

def convert(array):
    preprocessed_data = []
    
    for i in range(len(array)):
        item = array[i]
        bits = list(map(lambda bit: int(bit), list(item)))
        new_bits = []
        xor = bits[0]
        new_bits.append(xor)
        
        for bit in bits[1:]:
            xor ^= bit
            new_bits.append(xor)        
        preprocessed_data.append(new_bits)
    return np.array(preprocessed_data)


def lr(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def mlp(X, y):
    model = MLPClassifier(solver='sgd', max_iter=100, hidden_layer_sizes=(int(np.sqrt(X.shape[1]))))
    model.fit(X, y)
    return model

def boost(X, y):
    model = AdaBoostClassifier(n_estimators=100)
    model.fit(X, y)
    return model

N = list(range(8, 128, 8))
score = []
for i in N:
    print(i)
    data = pd.read_csv('Data/Base' + str(i) + '/Base' + str(i) + '_raw.txt', delimiter=',', header=None)
    data_X, data_y = data[data.columns[:-1]].values, data[data.columns[-1]].values
    data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_X, data_y, train_size=0.7)
    model = boost(data_X_train, data_y_train)
    score.append(model.score(data_X_test, data_y_test))
    
# plt.xlabel('bits N')
# plt.ylabel('score')
# plt.plot(N, score)
# plt.show();

def train_per_size(data, N):
    data_X, data_y = data[data.columns[:-1]].values, data[data.columns[-1]].values
    train_sizes = list(np.linspace(0.1, 0.99, 3))
    scores = []
    for size in train_sizes:
        print(size)
        data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_X, data_y, train_size=size)
        model = boost(data_X_train, data_y_train)
        scores.append(model.score(data_X_test, data_y_test))

    print(scores)
    plt.figure()
    plt.xlabel('data size')
    plt.ylabel('score')
    plt.plot(train_sizes, scores)
    plt.show()

i = 128
data = pd.read_csv('Data/Base' + str(i) + '/Base' + str(i) + '_raw.txt', delimiter=',', header=None)
train_per_size(data, i)