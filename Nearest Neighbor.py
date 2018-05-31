import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


x_1 = unpickle('cifar-10-batches-py/data_batch_1')
x_2 = unpickle('cifar-10-batches-py/data_batch_2')
x_3 = unpickle('cifar-10-batches-py/data_batch_3')
x_4 = unpickle('cifar-10-batches-py/data_batch_4')
x_5 = unpickle('cifar-10-batches-py/data_batch_5')
d_1 = x_1[b'data']
d_2 = x_2[b'data']
d_3 = x_3[b'data']
d_4 = x_4[b'data']
l_1 = x_1[b'labels']
l_2 = x_2[b'labels']
l_3 = x_3[b'labels']
l_4 = x_4[b'labels']
data_training = np.concatenate((d_1, d_2, d_3, d_4), axis=0)
label_training = np.concatenate((l_1, l_2, l_3, l_4), axis=0)
data_testing = x_5[b'data']
label_testing = x_5[b'labels']


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        num_test = 10
        a = 0
        Ypred = np.zeros(num_test, dtype=self.y.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.x.astype('int16') - x[i, :].astype('int16')), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.y[min_index]
        for i in range(num_test):
            if Ypred[i] == label_testing[i]:
                a += 1
        print("Our prediction is ", Ypred.tolist())
        print("The prediction accuracy is ", int(a/num_test * 100), "%")


nn = NearestNeighbor()
nn.train(data_training, label_training)
print('The True Classes are ', label_testing)
nn.predict(data_testing)

