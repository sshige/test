import numpy as np
import matplotlib.pyplot as plt

import chainer

class TrajectoryDataset(chainer.dataset.DatasetMixin):
# y = a sin(b x + c) + d

    def __init__(self,
                 n_data=1000, data_length=20,
                 input_start=0., input_end=2*np.pi,
             ):
        self.n_data = n_data
        self.input_start = input_start
        self.input_end = input_end
        self.data_length = data_length
        self.a_array = np.random.uniform(0., 1., self.n_data)
        self.b_array = np.random.uniform(0., 2., self.n_data)
        self.c_array = np.random.uniform(0., np.pi, self.n_data)
        self.d_array = np.random.uniform(1., 2., self.n_data)

    def __len__(self):
        return self.n_data

    def get_example(self, i):
        x = np.linspace(self.input_start, self.input_end, self.data_length)
        a = self.a_array[i]
        b = self.b_array[i]
        c = self.c_array[i]
        d = self.d_array[i]
        y = a * np.sin(b * x + c) + d
        return y

    def visualize_example(self, i):
        x = np.linspace(self.input_start, self.input_end, self.data_length)
        y = self.get_example(i)
        plt.plot(x, y)
        plt.pause(0.1)


if __name__ == '__main__':
    td = TrajectoryDataset()
    for i in range(len(td)):
        td.visualize_example(i)
