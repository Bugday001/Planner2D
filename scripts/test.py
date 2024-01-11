import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib.animation as animation
import time
import threading


def getQM(n_seg, n_order, ts):
    d_order = (n_order + 1) // 2
    Q = np.empty((0, n_order + 1, n_order + 1))
    M = np.empty((0, n_order + 1, n_order + 1))

    for k in range(1, n_seg + 1):
        # STEP 2.1: 计算第k段的Q_k
        Q_k = np.zeros((n_order + 1, n_order + 1))
        for i in range(d_order, n_order + 1):
            for j in range(d_order, n_order + 1):
                Q_k[i, j] = (
                        np.math.factorial(i) / np.math.factorial(i - d_order) *
                        np.math.factorial(j) / np.math.factorial(j - d_order) /
                        (i + j - n_order) * ts[k - 1] ** (i + j - n_order)
                )
        M_k = np.array([[1, 0, 0, 0, 0, 0],
                        [-5, 5, 0, 0, 0, 0],
                        [10, -20, 10, 0, 0, 0],
                        [-10, 30, -30, 10, 0, 0],
                        [5, -20, 30, -20, 5, 0],
                        [-1, 5, -10, 10, -5, 1]])  # 请实现getM函数
        if Q.shape[0] == 0:
            Q = Q_k
            M = M_k
        else:
            Q = np.block([[Q, np.zeros((Q.shape[0], Q_k.shape[1]))],
                          [np.zeros((Q_k.shape[0], Q.shape[1])), Q_k]])

            M = np.block([[M, np.zeros((M.shape[0], M_k.shape[1]))],
                          [np.zeros((M_k.shape[0], M.shape[1])), M_k]])

    return Q, M


# Q, M = getQM(1, 5, np.array([1]))
# print(Q, '\n', M, '\n', M.T)
# print(M.T@Q@M)

def fordef():
    # 创建数据
    for i in range(10):
        s = time.time()
        plt.clf()
        x = np.linspace(0, 10, 100)
        y = np.sin(x + np.random.rand())
        plt.plot(x, y)
        plt.draw()
        plt.pause(0.1)
        print(time.time() - s)


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import numpy as np
import time


class Test:
    def __init__(self):
        # 创建一个figure和axes对象
        plt.ion()
        self.fig, self.ax = plt.subplots()
        # 初始化一个空的线对象
        self.ax.set_xlim(-1, 11)
        self.ax.set_ylim(-1, 11)
        self.line, = self.ax.plot([], [])
        self.new_data_event = threading.Event()  # Event to signal new data
        self.data_lock = threading.Lock()  # Lock to protect data access
        self.data = None  # Shared data between threads
        self.animation = animation.FuncAnimation(self.fig, self.update, interval=50, blit=True)
        plt.show(block=True)

    def get_new_data(self):
        # 根据需要生成新数据
        new_data = np.random.rand(5)
        return new_data

    def update(self, frame):
        data = self.get_new_data()
        self.line.set_data(np.arange(len(data)), data)  # 更新线的数据
        return self.line,


if __name__ == "__main__":
    test_instance = Test()
    for i in range(100):
        print(122)
        time.sleep(1)
        # plt.draw()
        # plt.pause(0.0000001)

