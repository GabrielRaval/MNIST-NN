# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 23:30:19 2020

@author: Gabriel
"""
import random
import time
import gzip
import numpy as np
from matplotlib import pyplot as plt

class Network(object):
    def __init__(self, sizes):
        """`sizes` is a list of integers representing number of neurons in
        each layer."""
        np.random.seed(0)
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.W = [np.random.standard_normal((nrows, ncolumns))
                  for nrows, ncolumns in zip(sizes[:-1], sizes[1:])]
        self.b = [np.random.standard_normal((1, size)) for size in sizes[1:]]

    def MSE(self, y_actual, y_pred):
        return np.mean(np.sum((y_actual - y_pred)**2, -1))

    def SE(self, y_actual, y_pred):
        return (y_pred - y_actual)**2/2

    def SE_gradient(self, y_actual, y_pred):
        return (y_pred - y_actual)

    def sigmoid(self, z):
        return 1/(1+np.e**(-z))

    def sigmoid_gradient(self, z):
        return (1+np.e**(-z))**(-2)*np.e**(-z)

    def feedforward(self, x):
        """Returns the output of the network if `x` is the input."""
        self.z = []  # List of weighted inputs
        self.a = [x,]  # List of activations including input layer
        a_L = x
        for W_L, b_L in zip(self.W, self.b):
            z_L = np.dot(a_L, W_L) + b_L
            a_L = self.sigmoid(z_L)
            self.z.append(z_L)
            self.a.append(a_L)
        return a_L

    def backpropagate(self, a_L, y_batch):
        delta = self.SE_gradient(y_batch, a_L)\
              * self.sigmoid_gradient(self.z[-1])
        deltas = [delta,]

        # deltas.append([np.dot(deltas[-1], W_i.T)*self.sigmoid_gradient(z_j)
        #                for W_i, z_j in zip(self.W[1:][::-1],
        #                                    self.z[:-1][::-1])])

        for W_i, z_j in zip(self.W[1:][::-1], self.z[:-1][::-1]):
            deltas.append(np.dot(deltas[-1], W_i.T)*self.sigmoid_gradient(z_j))

        deltas.reverse()
        dC_dW = [np.swapaxes(np.multiply(self.a[:-1][i][np.newaxis].T,
                                              deltas[i]), 0, 1)
                      for i in range(len(deltas))]
        dC_db = deltas

        return dC_dW, dC_db


    def gradient_descent(self, dC_dW, dC_db, m, r):
        self.W = [self.W[L] - r/m*np.sum(dC_dW[L], 0)
                  for L in np.arange(len(self.sizes)-1)]
        self.b = [self.b[L] - r/m*np.sum(dC_db[L], 0)
                  for L in np.arange(len(self.sizes)-1)]

    def evaluate(self, y_pred, y_valid):
        isequal = np.equal(y_pred.argmax(1), y_valid.argmax(1))
        return sum(isequal)/len(isequal)*100

    def SGD(self, train_data, epochs, mini_batch_size, r, test_data=None):
        x_train, y_train = train_data
        if test_data: x_test, y_test = test_data
        indices = np.arange(len(x_train))
        random.seed(0)
        for epoch in np.arange(1, epochs+1):
            random.shuffle(indices)
            batch_indices = [indices[i:i+mini_batch_size]
                             for i in indices[::mini_batch_size]]

            for n, index in enumerate(batch_indices, start=1):
                x_batch, y_batch = x_train[index], y_train[index]
                a_L = self.feedforward(x_batch)
                dC_dW, dC_db = self.backpropagate(a_L, y_batch)
                self.gradient_descent(dC_dW, dC_db, mini_batch_size, r)

                if (n) % 1000 == 0:
                    print(f'Batch {n} complete.')

            if ((epoch % 1 == 0) and test_data) or (epoch == epochs):
                self.y_pred = self.feedforward(x_test)
                self.score = self.evaluate(self.y_pred, y_test)
                print(f'Epoch {epoch} complete.'
                      +f' Score = {round(self.score, 2)} %')

            else: print('Epoch {epoch} complete.')

def extract_mnist(path):
    f = gzip.open(path, mode='rb')
    f.seek(3)
    num_dims = int.from_bytes(f.read(1), 'big')
    if num_dims == 1:
        f.seek(8)
        data = np.array(list(f.read(-1)))
    else:
        n_img = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_columns = int.from_bytes(f.read(4), 'big')
        data = np.reshape(list(f.read(-1)), (n_img, n_rows, n_columns))
    f.close()
    return data

def plot_digit(image):
    return plt.contourf(image[::-1, :], cmap='binary', levels=256)

def onehot(labels):
    encoded = np.zeros((len(labels), 10))
    index = np.array([np.arange(len(labels)), labels])
    encoded[index[0], index[1]] = 1
    return encoded

# train_images_gz_path = 'C:/Users/Gabriel/Google Drive/'\
#                      + 'Data science and machine learning/'\
#                      + 'number recognition/data/'\
#                      + 'train-images-idx3-ubyte.gz'
# train_labels_gz_path = 'C:/Users/Gabriel/Google Drive/'\
#                      + 'Data science and machine learning/'\
#                      + 'number recognition/data/'\
#                      + 'train-labels-idx1-ubyte.gz'

# test_images_gz_path = 'C:/Users/Gabriel/Google Drive/'\
#                       + 'Data science and machine learning/'\
#                       + 'number recognition/data/'\
#                       + 't10k-images-idx3-ubyte.gz'
# test_labels_gz_path = 'C:/Users/Gabriel/Google Drive/'\
#                       + 'Data science and machine learning/'\
#                       + 'number recognition/data/'\
#                       + 't10k-labels-idx1-ubyte.gz'

train_labels = np.load('train_labels.npy')[:1000]
train_images = np.load('train_images.npy')[:1000]
# train_labels = np.load('train_labels.npy')
# train_images = np.load('train_images.npy')
test_labels = np.load('test_labels.npy')
test_images = np.load('test_images.npy')

train_valid_ratio = 1
train_index = int(len(train_labels)*train_valid_ratio)

x = np.reshape(train_images, (len(train_images), 28**2))/255
x_train, x_valid = x[:train_index], x[train_index:]
y = onehot(train_labels)
y_train, y_valid = y[:train_index], y[train_index:]

y_test = onehot(test_labels)
x_test = np.reshape(test_images, (len(test_images), 28**2))/255

hidden_layer_sizes = [100]
layer_sizes = np.array([x.shape[-1]]
                        + hidden_layer_sizes
                        + [y.shape[-1]])
# layer_sizes = np.linspace(10, 784, 3).astype(int)[::-1]

train_data = (x_train, y_train)
test_data = (x_test, y_test)

trainstart = time.perf_counter()
net = Network(layer_sizes)
net.SGD(train_data, 18, 10, 3, test_data)
trainend = time.perf_counter()
print(f'Training complete in {round(trainend-trainstart, 2)} seconds')




# for i in np.where(y_pred.argmax(1)!=y_test.argmax(1))[0][:20]:
#     plt.contourf(test_images[i][::-1, :], cmap='binary', levels=256)
#     plt.title(f'This is a {test_labels[i]}, '
#               + f'the network thought it was a {y_pred.argmax(1)[i]}.')
#     plt.show()
